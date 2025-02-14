__kernel void mul_matrices_float(__global float *mat1, __global float *mat2, __global float *result, int N) {
    const int TILE_SIZE = 16;
    int row = get_global_id(1);
    int col = get_global_id(0);
    
    float sum = 0.0;

    // Déclaration des matrices locales en début de fonction
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    if (N < TILE_SIZE && row < N && col < N) {
        // Cas des petites matrices : multiplication classique
        if (row < N && col < N) {
            for (int k = 0; k < N; k++) {
                sum += mat1[row * N + k] * mat2[k * N + col];
            }
            result[row * N + col] = sum;
        }
    } else {
        // Multiplication avec Tiling
        int localRow = get_local_id(1);
        int localCol = get_local_id(0);
        int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;  

        for (int t = 0; t < numTiles; ++t) {
            int tiledRow = row, tiledCol = t * TILE_SIZE + localCol;
            tileA[localRow][localCol] = (tiledRow < N && tiledCol < N) ? mat1[tiledRow * N + tiledCol] : 0.0;
            
            tiledRow = t * TILE_SIZE + localRow, tiledCol = col;
            tileB[localRow][localCol] = (tiledRow < N && tiledCol < N) ? mat2[tiledRow * N + tiledCol] : 0.0;

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += tileA[localRow][k] * tileB[k][localCol];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (row < N && col < N) {
            result[row * N + col] = sum;
        }
    }
}

__kernel void mul_matrices_int(__global int *mat1, __global int *mat2, __global int *result, int N) {
    const int TILE_SIZE = 16;
    int row = get_global_id(1);
    int col = get_global_id(0);
    
    int sum = 0;

    // Déclaration des matrices locales en début de fonction
    __local int tileA[TILE_SIZE][TILE_SIZE];
    __local int tileB[TILE_SIZE][TILE_SIZE];

    if (N < TILE_SIZE && row < N && col < N) {
        // Cas des petites matrices : multiplication classique
        if (row < N && col < N) {
            for (int k = 0; k < N; k++) {
                sum += mat1[row * N + k] * mat2[k * N + col];
            }
            result[row * N + col] = sum;
        }
    } else {
        // Multiplication avec Tiling
        int localRow = get_local_id(1);
        int localCol = get_local_id(0);
        int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;  

        for (int t = 0; t < numTiles; ++t) {
            int tiledRow = row, tiledCol = t * TILE_SIZE + localCol;
            tileA[localRow][localCol] = (tiledRow < N && tiledCol < N) ? mat1[tiledRow * N + tiledCol] : 0;
            
            tiledRow = t * TILE_SIZE + localRow, tiledCol = col;
            tileB[localRow][localCol] = (tiledRow < N && tiledCol < N) ? mat2[tiledRow * N + tiledCol] : 0;

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += tileA[localRow][k] * tileB[k][localCol];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (row < N && col < N) {
            result[row * N + col] = sum;
        }
    }
}
