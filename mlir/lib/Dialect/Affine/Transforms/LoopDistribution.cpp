//===----------------------------------------------------------------------===//
//
// This file implements affine distribution.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <map>
#include <memory>
#include <utility>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINELOOPDISTRIBUTION
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-loop-distribution"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::presburger;

namespace {

// Extract constant shifts from an affine map of the form
// (i,j,k)->(i+c1, j+c2, ...)
llvm::SmallVector<int64_t, 4> getAffineMapShifts(AffineMap map) {
  llvm::SmallVector<int64_t, 4> shifts;
  shifts.reserve(map.getNumResults());

  for (AffineExpr expr : map.getResults()) {
    // We expect each result to be an AffineDimExpr + (optional) constant
    int64_t shift = 0;
    if (auto bin = llvm::dyn_cast<AffineBinaryOpExpr>(expr)) {
      if (bin.getKind() == AffineExprKind::Add) {
        if (auto constExpr = dyn_cast<AffineConstantExpr>(bin.getRHS())) {
          shift = constExpr.getValue();
        }
      } else {
        static_assert("Not a shifted map in getAffineMapShifts");
      }
    }
    shifts.push_back(shift);
  }

  return shifts;
}

bool isShiftedIdentity(AffineMap map) {
  // Must have same number of dims and results
  if (map.getNumDims() != map.getNumResults())
    return false;

  if (map.getNumSymbols() > 0) {
    return false;
  }

  for (unsigned idx = 0; idx < map.getNumResults(); ++idx) {
    AffineExpr expr = map.getResult(idx);

    // Direct dimension (no shift)
    if (isa<AffineDimExpr>(expr)) {
      if (cast<AffineDimExpr>(expr).getPosition() != idx)
        return false;
      continue;
    }

    // Dimension + constant
    if (auto bin = dyn_cast<AffineBinaryOpExpr>(expr)) {
      if (bin.getKind() == AffineExprKind::Add) {
        if (auto dim = dyn_cast<AffineDimExpr>(bin.getLHS())) {
          if (dim.getPosition() != idx)
            return false;
          if (!isa<AffineConstantExpr>(bin.getRHS()))
            return false;
          continue;
        }
      }
    }

    // Anything else = not a shifted identity
    return false;
  }

  return true;
}

struct LoopDistribution
    : mlir::affine::impl::AffineLoopDistributionBase<LoopDistribution> {
  using StatementMap = std::map<Operation *, llvm::SmallVector<Operation *>>;
  using StatementList = llvm::SmallVector<Operation *>;
  using OpBoolMap = std::map<Operation *, bool>;
  void updateOperands(Operation *root, IRMapping &mapping);
  Operation *getSplitPoint(AffineForOp loop);
  void distributeForLoop(AffineForOp loop, IRRewriter &rewriter);
  std::pair<AffineForOp, AffineForOp>
  distributeForLoopBySplitPoint(AffineForOp loop, IRRewriter &rewriter,
                                Operation *splitPoint);
  bool isValidSplitPoint(AffineForOp loop, Operation *splitPoint);
  bool isDistributionProfitable(AffineForOp loop);
  void runOnOperation() override;
  // std::optional<mlir::AliasAnalysis> aa;
};

} // namespace

bool LoopDistribution::isDistributionProfitable(AffineForOp loop) {
  auto footprint = getMemoryFootprintBytes(loop);

  LLVM_DEBUG(llvm::dbgs() << "Checking distribution profitability for loop at "
                          << loop.getLoc() << "\n";);

  if (footprint.has_value()) {
    LLVM_DEBUG(llvm::dbgs() << "Estimated memory footprint (bytes): "
                            << footprint.value() << "\n";);

    bool profitable = footprint.value() > MinDistributionFootprint;
    LLVM_DEBUG(llvm::dbgs() << "Distribution is "
                            << (profitable ? "profitable" : "not profitable")
                            << " (MinDistributionFootprint = "
                            << MinDistributionFootprint << " bytes)\n";);
    return profitable;
  }
  LLVM_DEBUG(
      llvm::dbgs()
          << "Could not estimate footprint, defaulting to profitable\n";);
  return true;
}

bool LoopDistribution::isValidSplitPoint(AffineForOp loop,
                                         Operation *splitPoint) {
  mlir::AliasAnalysis &aa = getAnalysis<AliasAnalysis>();
  LLVM_DEBUG(llvm::dbgs() << "Checking split point: ";
             splitPoint->print(llvm::dbgs()); llvm::dbgs() << "\n";);

  if (llvm::isa<AffineYieldOp>(splitPoint)) {
    LLVM_DEBUG(llvm::dbgs() << "Split point is AffineYieldOp -> invalid\n";);
    return false;
  }

  llvm::SmallVector<Operation *> operations;
  for (auto &op : loop.getBody()->getOperations())
    operations.push_back(&op);

  llvm::SmallVector<Operation *> loop1, loop2;
  for (Operation *curr : operations) {
    if (curr->isBeforeInBlock(splitPoint) && curr != splitPoint)
      loop1.push_back(curr);
    else
      loop2.push_back(curr);
  }

  LLVM_DEBUG(llvm::dbgs() << "Loop1 ops: " << loop1.size()
                          << ", Loop2 ops: " << loop2.size() << "\n";);

  Value indVar = loop.getInductionVar();

  // collect defs
  llvm::DenseSet<Value> defsInLoop1;
  for (Operation *op : loop1)
    for (Value v : op->getResults())
      defsInLoop1.insert(v);

  for (Operation *op : loop2) {
    for (Value operand : op->getOperands()) {
      if (defsInLoop1.contains(operand)) {
        LLVM_DEBUG(llvm::dbgs()
                       << "Split invalid: loop2 depends on loop1 value ";
                   operand.print(llvm::dbgs()); llvm::dbgs() << "\n";);
        return false;
      }
    }
  }

  // mem dep checks
  for (Operation *firstLoopOp : loop1) {
    for (Operation *secondLoopOp : loop2) {
      llvm::SmallVector<AffineReadOpInterface> reads1, reads2;
      llvm::SmallVector<AffineWriteOpInterface> writes1, writes2;

      firstLoopOp->walk([&](AffineReadOpInterface r) { reads1.push_back(r); });
      secondLoopOp->walk([&](AffineReadOpInterface r) { reads2.push_back(r); });
      firstLoopOp->walk(
          [&](AffineWriteOpInterface w) { writes1.push_back(w); });
      secondLoopOp->walk(
          [&](AffineWriteOpInterface w) { writes2.push_back(w); });

      if (auto r = llvm::dyn_cast<AffineReadOpInterface>(firstLoopOp)) {
        reads1.push_back(r);
      }
      if (auto r = llvm::dyn_cast<AffineReadOpInterface>(secondLoopOp)) {
        reads2.push_back(r);
      }
      if (auto w = llvm::dyn_cast<AffineWriteOpInterface>(firstLoopOp)) {
        writes1.push_back(w);
      }
      if (auto w = llvm::dyn_cast<AffineWriteOpInterface>(secondLoopOp)) {
        writes2.push_back(w);
      }

      // === RAW ===
      for (auto access1 : writes1) {
        for (auto access2 : reads2) {
          if (access2.getMemRef() != access1.getMemRef()) {
            auto aliasResult =
                aa.alias(access2.getMemRef(), access1.getMemRef());
            if (aliasResult != AliasResult::NoAlias) {
              if (distributeMayAlias && aliasResult == AliasResult::MayAlias) {
                LLVM_DEBUG(llvm::dbgs() << "Warning: MemRefs may alias\n"
                                        << access2 << "\n"
                                        << access1 << "\n";);
                continue;
              }
              LLVM_DEBUG(llvm::dbgs()
                             << "MemRefs alias, can't check dependence\n"
                             << access2 << "\n"
                             << access1 << "\n";);
              LLVM_DEBUG(access2->dump(); access1->dump());
              LLVM_DEBUG(aliasResult.print(llvm::dbgs()));
              return false;
            }
            continue;
          }
          LLVM_DEBUG(llvm::dbgs() << "Checking RAW: read from ";
                     access2.getMemRef().print(llvm::dbgs());
                     llvm::dbgs() << " vs write from ";
                     access1.getMemRef().print(llvm::dbgs());
                     llvm::dbgs() << "\n";);

          AffineMap map1 = access2.getAffineMap();
          AffineMap map2 = access1.getAffineMap();
          if (!isShiftedIdentity(map1) || !isShiftedIdentity(map2)) {
            LLVM_DEBUG(llvm::dbgs()
                           << "Non-shifted-identity map -> cannot split\n";);
            return false;
          }

          auto operands1 = access2.getMapOperands();
          auto operands2 = access1.getMapOperands();
          if (!llvm::is_contained(operands1, indVar) ||
              !llvm::is_contained(operands2, indVar)) {
            LLVM_DEBUG(llvm::dbgs() << "No dependency as " << indVar
                                    << " not in one of the map operands");
            LLVM_DEBUG(map1.dump());
            LLVM_DEBUG(map2.dump());
            continue;
          }
          int idx1 = llvm::find(operands1, indVar) - operands1.begin();
          int idx2 = llvm::find(operands2, indVar) - operands2.begin();
          if (idx1 < 0 || idx1 >= (int)map1.getNumDims() || idx2 < 0 ||
              idx2 >= (int)map2.getNumDims())
            return false;

          int shift1 = getAffineMapShifts(map1)[idx1];
          int shift2 = getAffineMapShifts(map2)[idx2];
          LLVM_DEBUG(llvm::dbgs() << "RAW shifts: " << shift1 << " -> "
                                  << shift2 << "\n";);
          if (shift2 - shift1 < 0) {
            LLVM_DEBUG(llvm::dbgs()
                           << "RAW hazard detected -> cannot split\n";);
            return false;
          }
        }
      }

      // === WAW ===
      for (auto access1 : writes1) {
        for (auto access2 : writes2) {
          if (access2.getMemRef() != access1.getMemRef()) {
            auto aliasResult =
                aa.alias(access2.getMemRef(), access1.getMemRef());
            if (aliasResult != AliasResult::NoAlias) {
              if (distributeMayAlias && aliasResult == AliasResult::MayAlias) {
                LLVM_DEBUG(llvm::dbgs() << "Warning: MemRefs may alias\n"
                                        << access2 << "\n"
                                        << access1 << "\n";);
                continue;
              }
              LLVM_DEBUG(llvm::dbgs()
                             << "MemRefs alias, can't check dependence\n"
                             << access2 << "\n"
                             << access1 << "\n";);
              LLVM_DEBUG(access2->dump(); access1->dump());
              LLVM_DEBUG(aliasResult.print(llvm::dbgs()));
              return false;
            }
            continue;
          }
          LLVM_DEBUG(llvm::dbgs() << "Checking WAW: write from ";
                     access1.getMemRef().print(llvm::dbgs());
                     llvm::dbgs() << " vs write from ";
                     access2.getMemRef().print(llvm::dbgs());
                     llvm::dbgs() << "\n";);

          AffineMap map1 = access1.getAffineMap();
          AffineMap map2 = access2.getAffineMap();
          if (!isShiftedIdentity(map1) || !isShiftedIdentity(map2))
            return false;

          auto operands1 = access1.getMapOperands();
          auto operands2 = access2.getMapOperands();
          if (!llvm::is_contained(operands1, indVar) ||
              !llvm::is_contained(operands2, indVar)) {
            LLVM_DEBUG(llvm::dbgs() << "No dependency as " << indVar
                                    << " not in one of the map operands");
            LLVM_DEBUG(map1.dump());
            LLVM_DEBUG(map2.dump());
            continue;
          }
          int idx1 = llvm::find(operands1, indVar) - operands1.begin();
          int idx2 = llvm::find(operands2, indVar) - operands2.begin();
          int shift1 = getAffineMapShifts(map1)[idx1];
          int shift2 = getAffineMapShifts(map2)[idx2];
          LLVM_DEBUG(llvm::dbgs() << "WAW shifts: " << shift1 << " -> "
                                  << shift2 << "\n";);
          if (shift2 - shift1 < 0) {
            LLVM_DEBUG(llvm::dbgs()
                           << "WAW hazard detected -> cannot split\n";);
            return false;
          }
        }
      }

      // === WAR ===
      for (auto access1 : reads1) {
        for (auto access2 : writes2) {
          if (access1.getMemRef() != access2.getMemRef()) {
            if (access2.getMemRef() != access1.getMemRef()) {
              auto aliasResult =
                  aa.alias(access2.getMemRef(), access1.getMemRef());
              if (aliasResult != AliasResult::NoAlias) {
                if (distributeMayAlias &&
                    aliasResult == AliasResult::MayAlias) {
                  LLVM_DEBUG(llvm::dbgs() << "Warning: MemRefs may alias\n"
                                          << access2 << "\n"
                                          << access1 << "\n";);
                  continue;
                }
                LLVM_DEBUG(llvm::dbgs()
                               << "MemRefs alias, can't check dependence\n"
                               << access2 << "\n"
                               << access1 << "\n";);
                LLVM_DEBUG(access2->dump(); access1->dump());
                LLVM_DEBUG(aliasResult.print(llvm::dbgs()));
                return false;
              }
              continue;
            }
            continue;
          }

          LLVM_DEBUG(llvm::dbgs() << "Checking WAR: read from ";
                     access1.getMemRef().print(llvm::dbgs());
                     llvm::dbgs() << " vs write from ";
                     access2.getMemRef().print(llvm::dbgs());
                     llvm::dbgs() << "\n";);

          AffineMap map1 = access1.getAffineMap();
          AffineMap map2 = access2.getAffineMap();
          if (!isShiftedIdentity(map1) || !isShiftedIdentity(map2))
            return false;

          auto operands1 = access1.getMapOperands();
          auto operands2 = access2.getMapOperands();
          if (!llvm::is_contained(operands1, indVar) ||
              !llvm::is_contained(operands2, indVar)) {
            LLVM_DEBUG(llvm::dbgs() << "No dependency as " << indVar
                                    << " not in one of the map operands");
            LLVM_DEBUG(map1.dump());
            LLVM_DEBUG(map2.dump());
            continue;
          }
          int idx1 = llvm::find(operands1, indVar) - operands1.begin();
          int idx2 = llvm::find(operands2, indVar) - operands2.begin();
          int shift1 = getAffineMapShifts(map1)[idx1];
          int shift2 = getAffineMapShifts(map2)[idx2];
          LLVM_DEBUG(llvm::dbgs() << "WAR shifts: " << shift1 << " -> "
                                  << shift2 << "\n";);
          if (shift2 - shift1 < 0) {
            LLVM_DEBUG(llvm::dbgs()
                           << "WAR hazard detected -> cannot split\n";);
            return false;
          }
        }
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Valid split point found\n";);
  return true;
}

Operation *LoopDistribution::getSplitPoint(AffineForOp loop) {
  LLVM_DEBUG(llvm::dbgs() << "Looking for split point in loop at "
                          << loop.getLoc() << "\n";);

  Block *loopBody = loop.getBody();
  auto &ops = loopBody->getOperations();

  for (auto it = std::next(ops.begin()); it != ops.end(); ++it) {
    if (isValidSplitPoint(loop, &*it)) {
      LLVM_DEBUG(llvm::dbgs() << "Chose split point: "; it->print(llvm::dbgs());
                 llvm::dbgs() << "\n";);
      return &*it;
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "No valid split point found.\n";);
  return nullptr;
}

std::pair<AffineForOp, AffineForOp>
LoopDistribution::distributeForLoopBySplitPoint(AffineForOp loop,
                                                IRRewriter &rewriter,
                                                Operation *splitPoint) {
  LLVM_DEBUG(llvm::dbgs() << "Distributing loop at " << loop.getLoc()
                          << " with split point: ";
             splitPoint->print(llvm::dbgs()); llvm::dbgs() << "\n";);
  rewriter.setInsertionPoint(loop);
  auto newLoop = rewriter.create<AffineForOp>(
      loop.getLoc(), loop.getLowerBoundOperands(), loop.getLowerBoundMap(),
      loop.getUpperBoundOperands(), loop.getUpperBoundMap(),
      loop.getStep().getSExtValue());
  IRMapping mapping;
  Block *oldBody = loop.getBody();
  Block *newBody = newLoop.getBody();

  // Map IVs
  for (auto [oldArg, newArg] :
       llvm::zip(oldBody->getArguments(), newBody->getArguments()))
    mapping.map(oldArg, newArg);

  mapping.map(loop.getInductionVar(), newLoop.getInductionVar());

  rewriter.setInsertionPointToStart(newBody);

  // Collect ops to clone
  llvm::SmallDenseSet<Operation *> toClone;
  for (auto &op : oldBody->getOperations()) {
    if (op.isBeforeInBlock(splitPoint) && &op != splitPoint) {
      toClone.insert(&op);
    }
  }

  for (auto &op : oldBody->getOperations()) {
    if (!toClone.contains(&op))
      continue;
    rewriter.clone(op, mapping);
  }

  // Fix operands inside new loop
  newLoop->walk([&](Operation *cloned) {
    for (OpOperand &operand : cloned->getOpOperands())
      if (Value mapped = mapping.lookupOrNull(operand.get()))
        operand.set(mapped);
  });
  LLVM_DEBUG(llvm::dbgs() << "Created new loop before original\n";);
  LLVM_DEBUG(newLoop.dump());
  llvm::SmallVector<Operation *> toErase;

  for (auto &op : oldBody->getOperations()) {
    if (toClone.contains(&op))
      toErase.push_back(&op);
  }

  for (auto it = toErase.rbegin(); it != toErase.rend(); ++it) {
    Operation *op = *it;
    LLVM_DEBUG(llvm::dbgs() << "Erasing "; op->dump());
    op->erase();
  }

  return {newLoop, loop};
}

void LoopDistribution::distributeForLoop(AffineForOp loop,
                                         IRRewriter &rewriter) {
  if (loop->getNumResults() > 0)
    return;
  if (isDistributionProfitable(loop)) {
    // Split until no more split points are found
    while (Operation *splitPoint = getSplitPoint(loop)) {
      distributeForLoopBySplitPoint(loop, rewriter, splitPoint);
    }
  }
}

void LoopDistribution::runOnOperation() {
  Operation *op = getOperation();

  // aa.emplace(AliasAnalysis(op));
  // AliasAnalysis aa(op);
  // aa.alias(Value lhs, Value rhs)
  LLVM_DEBUG(
      llvm::dbgs() << "Running AffineLoopDistribution on function/module\n";);

  IRRewriter rewriter(op->getContext());
  llvm::SmallVector<AffineForOp> loops;
  op->walk<mlir::WalkOrder::PostOrder>([&](AffineForOp loop) {
    if (loop.getNumResults() == 0 && loop.getNumRegionIterArgs() == 0) {
      loops.push_back(loop);
    }
  });

  LLVM_DEBUG(llvm::dbgs() << "Found " << loops.size() << " candidate loops\n";);
  for (auto loop : loops) {
    LLVM_DEBUG(llvm::dbgs() << "Attempting distribution on loop: ";
               loop.print(llvm::dbgs()); llvm::dbgs() << "\n";);
    distributeForLoop(loop, rewriter);
  }
}

std::unique_ptr<Pass> mlir::affine::createAffineLoopDistributionPass() {
  return std::make_unique<LoopDistribution>();
}