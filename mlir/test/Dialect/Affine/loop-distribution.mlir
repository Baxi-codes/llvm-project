// RUN: mlir-opt -affine-loop-distribution %s | FileCheck %s

// CHECK-LABEL: func @distribute_for_locality
func.func @distribute_for_locality(%A: memref<?x?xf32>, %C: memref<?x?xf32>) {
  %c0 = arith.constant 1 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %A, %c0 : memref<?x?xf32>
  %N = memref.dim %A, %c1 : memref<?x?xf32>
  %B = memref.alloc(%M, %N) : memref<?x?xf32>
  %D = memref.alloc(%M, %N) : memref<?x?xf32>
  affine.for %i = 0 to %M {
    affine.for %j = 0 to %N {
      %v = affine.load %A[%i, %j] : memref<?x?xf32>
      %vd = arith.addf %v, %v : f32
      affine.store %vd, %B[%i, %j] : memref<?x?xf32>
      %w = affine.load %C[%i, %j] : memref<?x?xf32>
      %wd = arith.mulf %w, %w : f32
      affine.store %wd, %D[%i, %j] : memref<?x?xf32>
    }
  }
  // CHECK:      affine.for %{{.*}} = 0 to %{{.*}}
  // CHECK-NEXT:   affine.for %{{.*}} = 0 to %{{.*}}
  // CHECK-NEXT:     affine.load
  // CHECK-NEXT:     arith.addf
  // CHECK-NEXT:     affine.store
  // CHECK:      affine.for %{{.*}} = 0 to %{{.*}}
  // CHECK-NEXT:   affine.for %{{.*}} = 0 to %{{.*}}
  // CHECK-NEXT:     affine.load
  // CHECK-NEXT:     arith.mulf
  // CHECK-NEXT:     affine.store
  return
}

// CHECK-LABEL: func @distribute_for_parallelism
func.func @distribute_for_parallelism(%A: memref<1024xf32>, %B: memref<1024xf32>) {
  %C = memref.alloc() : memref<1024xf32>
  affine.for %i = 1 to 1024 {
    %v = affine.load %A[%i] : memref<1024xf32>
    %vd = arith.addf %v, %v : f32
    affine.store %vd, %B[%i] : memref<1024xf32>
    %x = affine.load %B[%i - 1] : memref<1024xf32>
    affine.store %x, %C[%i] : memref<1024xf32>
  }
  // CHECK:      affine.for %{{.*}} = 1 to 1024
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT:   arith.addf
  // CHECK-NEXT:   affine.store
  // CHECK:      affine.for %{{.*}} = 1 to 1024
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT:   affine.store
  return
}

// CHECK-LABEL: func @do_not_distribute_alias
func.func @do_not_distribute_alias(%A: memref<1024xf32>, %B: memref<1024xf32>) {
  // %C aliases with %A.
  %C = memref.reinterpret_cast %A to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<1024xf32> to memref<32x32xf32>
  affine.for %i = 1 to 1024 {
    %v = affine.load %A[%i] : memref<1024xf32>
    %vd = arith.addf %v, %v : f32
    affine.store %vd, %B[%i] : memref<1024xf32>
    %x = affine.load %B[%i - 1] : memref<1024xf32>
    affine.store %x, %C[%i mod 32, %i floordiv 32] : memref<32x32xf32>
  }
  // CHECK:      affine.for %{{.*}} = 1 to 1024
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT:   arith.addf
  // CHECK-NEXT:   affine.store
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT:   affine.store
  return
}

// CHECK-LABEL: func @incorrect_distribution
func.func @incorrect_distribution(%A: memref<1024xf32>, %B: memref<1024xf32>) {
  %C = memref.alloc() : memref<1024xf32>
  affine.for %i = 1 to 1023 {
    %v = affine.load %A[%i] : memref<1024xf32>
    %vd = arith.addf %v, %v : f32
    affine.store %vd, %B[%i] : memref<1024xf32>
    %x = affine.load %B[%i + 1] : memref<1024xf32>
    affine.store %x, %C[%i] : memref<1024xf32>
  }
  // CHECK:      affine.for %{{.*}} = 1 to 1023
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT:   arith.addf
  // CHECK-NEXT:   affine.store
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT:   affine.store
  return
}

// CHECK-LABEL: func @only_distribute_inner
func.func @only_distribute_inner(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) {
  %c_1 = arith.constant 1.0 : f32
  affine.for %i = 0 to 1023 {
    affine.for %j = 1 to 1023 {
      %a = affine.load %A[%i+1, %j] : memref<1024x1024xf32>
      %sum = arith.addf %a, %c_1 : f32
      affine.store %sum, %B[%i, %j] : memref<1024x1024xf32>
      %b = affine.load %B[%i, %j - 1] : memref<1024x1024xf32>
      affine.store %b, %A[%i,%j] :  memref<1024x1024xf32>
    }
  }
  // CHECK:      affine.for %{{.*}} = 0 to 1023
  // CHECK-NEXT:   affine.for %{{.*}} = 1 to 1023
  // CHECK-NEXT:     affine.load
  // CHECK-NEXT:     arith.addf
  // CHECK-NEXT:     affine.store
  // CHECK:      affine.for %{{.*}} = 1 to 1023
  // CHECK-NEXT:     affine.load
  // CHECK-NEXT:     affine.store
  return
}

// CHECK-LABEL: func @non_constant_distance
func.func @non_constant_distance(%A: memref<1024xf32>, %B: memref<1024xf32>) {
  %C = memref.alloc() : memref<1024xf32>
  affine.for %i = 1 to 1023 {
    %v = affine.load %A[%i] : memref<1024xf32>
    %vd = arith.addf %v, %v : f32
    affine.store %vd, %B[%i] : memref<1024xf32>
    %x = affine.load %B[%i*2 + 1] : memref<1024xf32>
    affine.store %x, %C[%i] : memref<1024xf32>
  }
  // CHECK:      affine.for %{{.*}} = 1 to 1023
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT:   arith.addf
  // CHECK-NEXT:   affine.store
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT:   affine.store
  return
}

// CHECK-LABEL: func @too_low_memory_footprint
func.func @too_low_memory_footprint(%A: memref<1024xf32>, %B: memref<1024xf32>) {
  %C = memref.alloc() : memref<1024xf32>
  affine.for %i = 1 to 6 {
    %v = affine.load %A[%i] : memref<1024xf32>
    %vd = arith.addf %v, %v : f32
    affine.store %vd, %B[%i] : memref<1024xf32>
    %x = affine.load %B[%i - 1] : memref<1024xf32>
    affine.store %x, %C[%i] : memref<1024xf32>
  }
  // CHECK:      affine.for %{{.*}} = 1 to 6
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT:   arith.addf
  // CHECK-NEXT:   affine.store
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT:   affine.store
  return
}

// CHECK-LABEL: func @distribute_imperfect
func.func @distribute_imperfect(%A: memref<32x32xf32>, %B: memref<32x32xf32>) {
  affine.for %i = 0 to 32 {
    %sum = arith.constant 0.0 : f32
    %tmp = affine.load %A[%i, 0] : memref<32x32xf32>
    %sum2 = arith.addf %sum, %tmp : f32
    affine.for %j = 0 to 32 {
      %v = affine.load %A[%i, %j] : memref<32x32xf32>
      %vd = arith.addf %v, %v : f32
      affine.store %vd, %B[%i, %j] : memref<32x32xf32>
    }
  }
  // CHECK:      affine.for %{{.*}} = 0 to 32
  // CHECK-NEXT:   arith.constant
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT:   arith.addf
  // CHECK:      affine.for %{{.*}} = 0 to 32
  // CHECK-NEXT:   affine.for %{{.*}} = 0 to 32
  // CHECK-NEXT:     affine.load
  // CHECK-NEXT:     arith.addf
  // CHECK-NEXT:     affine.store
  return
}

func.func @if_else(%A: memref<2048x2048xf64>) {
  %c0 = arith.constant 0.0 : f64
  %c1 = arith.constant 1.0 : f64
  affine.for %i = 0 to 2048 {
    affine.if affine_set<(d0) : (d0 - 1024 >= 0)> (%i) {
      affine.for %j = 0 to 2048 {
        affine.store %c0, %A[%i, %j] : memref<2048x2048xf64>
      }
    } else {
      affine.for %j = 0 to 2048 {
        affine.store %c1, %A[%i, %j] : memref<2048x2048xf64>
      }
    }
  }
  return
}
