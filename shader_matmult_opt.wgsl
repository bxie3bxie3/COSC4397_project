struct MatrixWSizeHeader {
    dim: vec2f,
    data: array<f32>,
}

@group(0) @binding(0) var<storage, read> left_matrix: MatrixWSizeHeader;
@group(0) @binding(1) var<storage, read> right_matrix: MatrixWSizeHeader;
@group(0) @binding(2) var<storage, read_write> result_matrix: MatrixWSizeHeader;


const BLOCKSIZE: u32 = 16;
const TILE_M: u32 = 8;  // Tile size in M dimension
const TILE_N: u32 = 8;  // Tile size in N dimension

@compute @workgroup_size(BLOCKSIZE, BLOCKSIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let left_matrix_rows = u32(left_matrix.dim.x);
	let right_matrix_cols = u32(right_matrix.dim.y);
	let matrices_inner_dim = u32(left_matrix.dim.y);
    let tile_start_row = global_id.y * TILE_M;
    let tile_start_col = global_id.x * TILE_N;

    var tile_result: array<array<f32, TILE_N>, TILE_M>;
    for (var row = 0u; row < TILE_M; row++) {
        for (var col = 0u; col < TILE_N; col++) {
            tile_result[row][col] = 0.0;
        }
    }

    // Compute the 2D tile
    for (var inner_index = 0u; inner_index < matrices_inner_dim; inner_index++) {
      let left_00 = left_matrix.data[tile_start_row * matrices_inner_dim + inner_index];
      let left01 = left_matrix.data[(tile_start_row + 1) * matrices_inner_dim + inner_index];
      let left02 = left_matrix.data[(tile_start_row + 2) * matrices_inner_dim + inner_index];
      let left03 = left_matrix.data[(tile_start_row + 3) * matrices_inner_dim + inner_index];
      let left04 = left_matrix.data[(tile_start_row + 4) * matrices_inner_dim + inner_index];
      let left05 = left_matrix.data[(tile_start_row + 5) * matrices_inner_dim + inner_index];
      let left06 = left_matrix.data[(tile_start_row + 6) * matrices_inner_dim + inner_index];
      let left07 = left_matrix.data[(tile_start_row + 7) * matrices_inner_dim + inner_index];
      let right_00 = right_matrix.data[inner_index * right_matrix_cols + tile_start_col];
      let right01 = right_matrix.data[inner_index * right_matrix_cols + (tile_start_col + 1)];
      let right02 = right_matrix.data[inner_index * right_matrix_cols + (tile_start_col + 2)];
      let right03 = right_matrix.data[inner_index * right_matrix_cols + (tile_start_col + 3)];
      let right04 = right_matrix.data[inner_index * right_matrix_cols + (tile_start_col + 4)];
      let right05 = right_matrix.data[inner_index * right_matrix_cols + (tile_start_col + 5)];
      let right06 = right_matrix.data[inner_index * right_matrix_cols + (tile_start_col + 6)];
      let right07 = right_matrix.data[inner_index * right_matrix_cols + (tile_start_col + 7)];
      tile_result[0][0] += left_00 * right_00;
      tile_result[0][1] += left_00 * right01;
      tile_result[0][2] += left_00 * right02;
      tile_result[0][3] += left_00 * right03;
      tile_result[0][4] += left_00 * right04;
      tile_result[0][5] += left_00 * right05;
      tile_result[0][6] += left_00 * right06;
      tile_result[0][7] += left_00 * right07;
      tile_result[1][0] += left01  * right_00;
      tile_result[1][1] += left01  * right01;
      tile_result[1][2] += left01  * right02;
      tile_result[1][3] += left01  * right03;
      tile_result[1][4] += left01  * right04;
      tile_result[1][5] += left01  * right05;
      tile_result[1][6] += left01  * right06;
      tile_result[1][7] += left01  * right07;
      tile_result[2][0] += left02  * right_00;
      tile_result[2][1] += left02  * right01;
      tile_result[2][2] += left02  * right02;
      tile_result[2][3] += left02  * right03;
      tile_result[2][4] += left02  * right04;
      tile_result[2][5] += left02  * right05;
      tile_result[2][6] += left02  * right06;
      tile_result[2][7] += left02  * right07;
      tile_result[3][0] += left03  * right_00;
      tile_result[3][1] += left03  * right01;
      tile_result[3][2] += left03  * right02;
      tile_result[3][3] += left03  * right03;
      tile_result[3][4] += left03  * right04;
      tile_result[3][5] += left03  * right05;
      tile_result[3][6] += left03  * right06;
      tile_result[3][7] += left03  * right07;
      tile_result[4][0] += left04  * right_00;
      tile_result[4][1] += left04  * right01;
      tile_result[4][2] += left04  * right02;
      tile_result[4][3] += left04  * right03;
      tile_result[4][4] += left04  * right04;
      tile_result[4][5] += left04  * right05;
      tile_result[4][6] += left04  * right06;
      tile_result[4][7] += left04  * right07;
      tile_result[5][0] += left05  * right_00;
      tile_result[5][1] += left05  * right01;
      tile_result[5][2] += left05  * right02;
      tile_result[5][3] += left05  * right03;
      tile_result[5][4] += left05  * right04;
      tile_result[5][5] += left05  * right05;
      tile_result[5][6] += left05  * right06;
      tile_result[5][7] += left05  * right07;
      tile_result[6][0] += left06  * right_00;
      tile_result[6][1] += left06  * right01;
      tile_result[6][2] += left06  * right02;
      tile_result[6][3] += left06  * right03;
      tile_result[6][4] += left06  * right04;
      tile_result[6][5] += left06  * right05;
      tile_result[6][6] += left06  * right06;
      tile_result[6][7] += left06  * right07;
      tile_result[7][0] += left07  * right_00;
      tile_result[7][1] += left07  * right01;
      tile_result[7][2] += left07  * right02;
      tile_result[7][3] += left07  * right03;
      tile_result[7][4] += left07  * right04;
      tile_result[7][5] += left07  * right05;
      tile_result[7][6] += left07  * right06;
      tile_result[7][7] += left07  * right07;
    }
    result_matrix.data[tile_start_row * right_matrix_cols + tile_start_col] = tile_result[0][0];
    result_matrix.data[tile_start_row * right_matrix_cols + (tile_start_col + 1)] = tile_result[0][1];
    result_matrix.data[tile_start_row * right_matrix_cols + (tile_start_col + 2)] = tile_result[0][2];
    result_matrix.data[tile_start_row * right_matrix_cols + (tile_start_col + 3)] = tile_result[0][3];
    result_matrix.data[tile_start_row * right_matrix_cols + (tile_start_col + 4)] = tile_result[0][4];
    result_matrix.data[tile_start_row * right_matrix_cols + (tile_start_col + 5)] = tile_result[0][5];
    result_matrix.data[tile_start_row * right_matrix_cols + (tile_start_col + 6)] = tile_result[0][6];
    result_matrix.data[tile_start_row * right_matrix_cols + (tile_start_col + 7)] = tile_result[0][7];
    result_matrix.data[(tile_start_row + 1) * right_matrix_cols + tile_start_col] = tile_result[1][0];
    result_matrix.data[(tile_start_row + 1) * right_matrix_cols + (tile_start_col + 1)] = tile_result[1][1];
    result_matrix.data[(tile_start_row + 1) * right_matrix_cols + (tile_start_col + 2)] = tile_result[1][2];
    result_matrix.data[(tile_start_row + 1) * right_matrix_cols + (tile_start_col + 3)] = tile_result[1][3];
    result_matrix.data[(tile_start_row + 1) * right_matrix_cols + (tile_start_col + 4)] = tile_result[1][4];
    result_matrix.data[(tile_start_row + 1) * right_matrix_cols + (tile_start_col + 5)] = tile_result[1][5];
    result_matrix.data[(tile_start_row + 1) * right_matrix_cols + (tile_start_col + 6)] = tile_result[1][6];
    result_matrix.data[(tile_start_row + 1) * right_matrix_cols + (tile_start_col + 7)] = tile_result[1][7];
    result_matrix.data[(tile_start_row + 2) * right_matrix_cols + tile_start_col] = tile_result[2][0];
    result_matrix.data[(tile_start_row + 2) * right_matrix_cols + (tile_start_col + 1)] = tile_result[2][1];
    result_matrix.data[(tile_start_row + 2) * right_matrix_cols + (tile_start_col + 2)] = tile_result[2][2];
    result_matrix.data[(tile_start_row + 2) * right_matrix_cols + (tile_start_col + 3)] = tile_result[2][3];
    result_matrix.data[(tile_start_row + 2) * right_matrix_cols + (tile_start_col + 4)] = tile_result[2][4];
    result_matrix.data[(tile_start_row + 2) * right_matrix_cols + (tile_start_col + 5)] = tile_result[2][5];
    result_matrix.data[(tile_start_row + 2) * right_matrix_cols + (tile_start_col + 6)] = tile_result[2][6];
    result_matrix.data[(tile_start_row + 2) * right_matrix_cols + (tile_start_col + 7)] = tile_result[2][7];
    result_matrix.data[(tile_start_row + 3) * right_matrix_cols + tile_start_col] = tile_result[3][0];
    result_matrix.data[(tile_start_row + 3) * right_matrix_cols + (tile_start_col + 1)] = tile_result[3][1];
    result_matrix.data[(tile_start_row + 3) * right_matrix_cols + (tile_start_col + 2)] = tile_result[3][2];
    result_matrix.data[(tile_start_row + 3) * right_matrix_cols + (tile_start_col + 3)] = tile_result[3][3];
    result_matrix.data[(tile_start_row + 3) * right_matrix_cols + (tile_start_col + 4)] = tile_result[3][4];
    result_matrix.data[(tile_start_row + 3) * right_matrix_cols + (tile_start_col + 5)] = tile_result[3][5];
    result_matrix.data[(tile_start_row + 3) * right_matrix_cols + (tile_start_col + 6)] = tile_result[3][6];
    result_matrix.data[(tile_start_row + 3) * right_matrix_cols + (tile_start_col + 7)] = tile_result[3][7];
    result_matrix.data[(tile_start_row + 4) * right_matrix_cols + tile_start_col] = tile_result[4][0];
    result_matrix.data[(tile_start_row + 4) * right_matrix_cols + (tile_start_col + 1)] = tile_result[4][1];
    result_matrix.data[(tile_start_row + 4) * right_matrix_cols + (tile_start_col + 2)] = tile_result[4][2];
    result_matrix.data[(tile_start_row + 4) * right_matrix_cols + (tile_start_col + 3)] = tile_result[4][3];
    result_matrix.data[(tile_start_row + 4) * right_matrix_cols + (tile_start_col + 4)] = tile_result[4][4];
    result_matrix.data[(tile_start_row + 4) * right_matrix_cols + (tile_start_col + 5)] = tile_result[4][5];
    result_matrix.data[(tile_start_row + 4) * right_matrix_cols + (tile_start_col + 6)] = tile_result[4][6];
    result_matrix.data[(tile_start_row + 4) * right_matrix_cols + (tile_start_col + 7)] = tile_result[4][7];
    result_matrix.data[(tile_start_row + 5) * right_matrix_cols + tile_start_col] = tile_result[5][0];
    result_matrix.data[(tile_start_row + 5) * right_matrix_cols + (tile_start_col + 1)] = tile_result[5][1];
    result_matrix.data[(tile_start_row + 5) * right_matrix_cols + (tile_start_col + 2)] = tile_result[5][2];
    result_matrix.data[(tile_start_row + 5) * right_matrix_cols + (tile_start_col + 3)] = tile_result[5][3];
    result_matrix.data[(tile_start_row + 5) * right_matrix_cols + (tile_start_col + 4)] = tile_result[5][4];
    result_matrix.data[(tile_start_row + 5) * right_matrix_cols + (tile_start_col + 5)] = tile_result[5][5];
    result_matrix.data[(tile_start_row + 5) * right_matrix_cols + (tile_start_col + 6)] = tile_result[5][6];
    result_matrix.data[(tile_start_row + 5) * right_matrix_cols + (tile_start_col + 7)] = tile_result[5][7];
    result_matrix.data[(tile_start_row + 6) * right_matrix_cols + tile_start_col] = tile_result[6][0];
    result_matrix.data[(tile_start_row + 6) * right_matrix_cols + (tile_start_col + 1)] = tile_result[6][1];
    result_matrix.data[(tile_start_row + 6) * right_matrix_cols + (tile_start_col + 2)] = tile_result[6][2];
    result_matrix.data[(tile_start_row + 6) * right_matrix_cols + (tile_start_col + 3)] = tile_result[6][3];
    result_matrix.data[(tile_start_row + 6) * right_matrix_cols + (tile_start_col + 4)] = tile_result[6][4];
    result_matrix.data[(tile_start_row + 6) * right_matrix_cols + (tile_start_col + 5)] = tile_result[6][5];
    result_matrix.data[(tile_start_row + 6) * right_matrix_cols + (tile_start_col + 6)] = tile_result[6][6];
    result_matrix.data[(tile_start_row + 6) * right_matrix_cols + (tile_start_col + 7)] = tile_result[6][7];
    result_matrix.data[(tile_start_row + 7) * right_matrix_cols + tile_start_col] = tile_result[7][0];
    result_matrix.data[(tile_start_row + 7) * right_matrix_cols + (tile_start_col + 1)] = tile_result[7][1];
    result_matrix.data[(tile_start_row + 7) * right_matrix_cols + (tile_start_col + 2)] = tile_result[7][2];
    result_matrix.data[(tile_start_row + 7) * right_matrix_cols + (tile_start_col + 3)] = tile_result[7][3];
    result_matrix.data[(tile_start_row + 7) * right_matrix_cols + (tile_start_col + 4)] = tile_result[7][4];
    result_matrix.data[(tile_start_row + 7) * right_matrix_cols + (tile_start_col + 5)] = tile_result[7][5];
    result_matrix.data[(tile_start_row + 7) * right_matrix_cols + (tile_start_col + 6)] = tile_result[7][6];
    result_matrix.data[(tile_start_row + 7) * right_matrix_cols + (tile_start_col + 7)] = tile_result[7][7];

}
