// kernel.cl

/*
* 특징:
* 1. 16채널씩 병렬처리 (__local float4 local_filter[48])
* 2. 3x3 필터를 4+4+1로 분할해서 SIMD 연산
* 3. Local memory에 필터 저장
*
* 메모리 접근 최적화:
* [필터 메모리]     →    [Local Memory]    →    [계산]
* Global Memory            Work-group 내         dot product로
*                         공유 메모리 사용       벡터화 연산
*/
__kernel void convolution_large(
	__global float* input,
	__global float* output,
	__global float* filter,
	__constant float* biases,
	const int inDim,
	const int outDim,
	const int nbyn) {

	__local float4 local_filter[48];

	int col = get_global_id(0);
	int batch_out_row = get_global_id(1);
	int local_x = get_local_id(0);
	int local_y = get_local_id(1);

	int batch_idx = batch_out_row / (outDim * nbyn);
	int channel_row_idx = batch_out_row % (outDim * nbyn);
	int out_channel = channel_row_idx / nbyn;
	int row = channel_row_idx % nbyn;
	int feature_map_size = nbyn * nbyn;
	float sum = 0.0f;

	for (int in_channel_base = 0; in_channel_base < inDim; in_channel_base += 16) {
		int local_id = local_y * get_local_size(0) + local_x;
		if (local_id < 48) {
			int ch_idx = local_id / 3;

			/* vload4(offset, pointer)
			* float arr[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
			* float4 result = vload4(0, arr);  ->  result = (1.0, 2.0, 3.0, 4.0)
			* float4 result2 = vload4(1, arr); ->  result2 = (2.0, 3.0, 4.0, 5.0)
			*/
			if (in_channel_base + ch_idx < inDim) {
				int filter_offset = (out_channel * inDim + in_channel_base + ch_idx) * 9;
				local_filter[local_id] = vload4(local_id % 3, filter + filter_offset);
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int ch = 0; ch < 16 && in_channel_base + ch < inDim; ch++) {
			float4 input_vec0 = 0.0f;
			float4 input_vec1 = 0.0f;
			float last_input = 0.0f;

			int input_base = batch_idx * inDim * feature_map_size +
				(in_channel_base + ch) * feature_map_size;

			int base_y = row - 1;
			int base_x = col - 1;

			for (int filter_row = 0; filter_row < 3; filter_row++) {
				int y = base_y + filter_row;
				if (y >= 0 && y < nbyn) {
					int row_offset = y * nbyn;
					for (int filter_col = 0; filter_col < 3; filter_col++) {
						int x = base_x + filter_col;
						if (x >= 0 && x < nbyn) {
							float val = input[input_base + row_offset + x];
							int kernel_idx = filter_row * 3 + filter_col;
							if (kernel_idx < 4) input_vec0[kernel_idx] = val;
							else if (kernel_idx < 8) input_vec1[kernel_idx - 4] = val;
							else last_input = val;
						}
					}
				}
			}

			float4 filter_vec0 = local_filter[ch * 3];
			float4 filter_vec1 = local_filter[ch * 3 + 1];
			float last_filter = local_filter[ch * 3 + 2].x;

			/* dot(vector1, vector2)
			* float4 a = (float4)(1,2,3,4);
			* float4 b = (float4)(5,6,7,8);
			* float result = dot(a,b);      ->  result = 1*5 + 2*6 + 3*7 + 4*8 = 70
			*/
			sum += dot(input_vec0, filter_vec0);
			sum += dot(input_vec1, filter_vec1);
			sum += last_input * last_filter;

			//sum = fma(input_vec0.x, filter_vec0.x, sum);
			//sum = fma(input_vec0.y, filter_vec0.y, sum);
			//sum = fma(input_vec0.z, filter_vec0.z, sum);
			//sum = fma(input_vec0.w, filter_vec0.w, sum);

			//sum = fma(input_vec1.x, filter_vec1.x, sum);
			//sum = fma(input_vec1.y, filter_vec1.y, sum);
			//sum = fma(input_vec1.z, filter_vec1.z, sum);
			//sum = fma(input_vec1.w, filter_vec1.w, sum);

			//sum = fma(last_input, last_filter, sum);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	/* fmax(x, y)
	* float a = fmax(3.14f, 2.71f);  ->  a = 3.14
	* float4 result = fmax(vec1, 0.0f);  ->  각 요소별로 0과 비교하여 큰 값 선택
	*/
	sum += biases[out_channel];
	output[batch_idx * outDim * feature_map_size +
		out_channel * feature_map_size +
		row * nbyn + col] = fmax(0.0f, sum);
}

/*
* 특징: 한 행씩 벡터화해서 처리
* 입력 행렬 / 필터에서 한 행씩 처리:
* [a b c]     →    input_vec = (a,b,c,0) / filter_vec = (1,2,3,0)
* [d e f]          각 행마다 float4로 로드
* [g h i]
*/

__kernel void convolution_small(
	__global float* input,
	__global float* output,
	__global float* filter,
	__constant float* biases,
	const int inDim,
	const int outDim,
	const int nbyn) {

	int col = get_global_id(0);
	int batch_out_row = get_global_id(1);

	int batch_idx = batch_out_row / (outDim * nbyn);
	int channel_row_idx = batch_out_row % (outDim * nbyn);
	int out_channel = channel_row_idx / nbyn;
	int row = channel_row_idx % nbyn;
	int feature_map_size = nbyn * nbyn;
	float4 sub_sum = (float4)0.0f;

	for (int in_channel = 0; in_channel < inDim; in_channel++) {
		int input_offset = (batch_idx * inDim + in_channel) * feature_map_size;
		int filter_offset = (out_channel * inDim + in_channel) * 9;
		int base_x = col - 1;

		for (int filter_row = 0; filter_row < 3; filter_row++) {
			int base_y = row + filter_row - 1;
			if (base_y >= 0 && base_y < nbyn) {
				float4 input_vec = (float4)(
					(base_x >= 0 && base_x < nbyn) ? input[input_offset + base_y * nbyn + base_x] : 0.0f,
					(base_x + 1 >= 0 && base_x + 1 < nbyn) ? input[input_offset + base_y * nbyn + base_x + 1] : 0.0f,
					(base_x + 2 >= 0 && base_x + 2 < nbyn) ? input[input_offset + base_y * nbyn + base_x + 2] : 0.0f,
					0.0f
					);

				/* any(vector)
				* float4 vec = (float4)(0.0f, 0.0f, 1.0f, 0.0f);
				* if(any(vec != 0.0f)) {  ->  true 반환 (세 번째 요소가 0이 아님)
				* 실행될 코드
				* }
				*
				* float4 zeros = (float4)(0.0f);
				* if(any(vec != zeros)) {  ->  벡터 전체 비교도 가능
				* 실행될 코드
				* }
				*/

				if (any(input_vec != (float4)(0.0f))) {
					float4 filter_vec = vload4(0, filter + filter_offset + filter_row * 3);
					sub_sum += (input_vec * filter_vec);
				}
			}
		}
	}

	float final_sum = sub_sum.x + sub_sum.y + sub_sum.z + biases[out_channel];
	output[batch_idx * outDim * feature_map_size +
		out_channel * feature_map_size + row * nbyn + col] = fmax(0.0f, final_sum);
}

/*
* 2x2 입력을       →       최대값 선택
* [a b]                    max(a,b,c,d)
* [c d]
*/
__kernel void max_pooling(
	__global float* input,
	__global float* output,
	const int nbyn,
	const int outDim) {

	int col = get_global_id(0);
	int batch_out_row = get_global_id(1);

	int batch_idx = batch_out_row / (outDim * nbyn);
	int channel_row_idx = batch_out_row % (outDim * nbyn);
	int out_channel = channel_row_idx / nbyn;
	int row = channel_row_idx % nbyn;
	int input_feature_map_size = (nbyn * 2) * (nbyn * 2);
	int output_feature_map_size = nbyn * nbyn;

	int input_row = row * 2;
	int input_col = col * 2;
	int input_start = batch_idx * outDim * input_feature_map_size +
		out_channel * input_feature_map_size +
		input_row * nbyn * 2 + input_col;

	/* vload2(offset, pointer)
	* float arr[4] = {1.0, 2.0, 3.0, 4.0};
	* float2 result = vload2(0, arr);  ->  result = (1.0, 2.0)
	* float2 result2 = vload2(1, arr); -> result2 = (2.0, 3.0)
	*/
	float4 block;
	block.xy = vload2(0, &input[input_start]);
	block.zw = vload2(0, &input[input_start + nbyn * 2]);

	/* max(vector1, vector2)
	* float2 a = (float2)(1.0, 4.0);
	* float2 b = (float2)(2.0, 3.0);
	* float2 result = max(a, b);  ->  result = (2.0, 4.0)
	*/
	float2 max_rows = max(block.xy, block.zw);
	output[batch_idx * outDim * output_feature_map_size +
		out_channel * output_feature_map_size +
		row * nbyn + col] = max(max_rows.x, max_rows.y);
}

/*
* 입력 벡터와 가중치 행렬의 내적:
* [a b c d] · [w1]    →    a*w1 + b*w2 + c*w3 + d*w4
*              [w2]
*              [w3]
*              [w4]
*/
__kernel void fc_layer(
	__global float* input,
	__global float* output,
	__global float* weights,
	__constant float* biases,
	const int inDim,
	const int outDim) {

	int out_channel = get_global_id(0);
	int batch_idx = get_global_id(1);

	int batch_offset = batch_idx * inDim;
	int weight_offset = out_channel * inDim;
	float sum = 0.0f;

	for (int i = 0; i < inDim; i += 4) {
		float4 input_vec = vload4(0, input + batch_offset + i);
		float4 weight_vec = vload4(0, weights + weight_offset + i);
		sum += dot(input_vec, weight_vec);
		//sum = fma(input_vec.x, weight_vec.x, sum);
		//sum = fma(input_vec.y, weight_vec.y, sum);
		//sum = fma(input_vec.z, weight_vec.z, sum);
		//sum = fma(input_vec.w, weight_vec.w, sum);
	}

	sum += biases[out_channel];
	output[batch_idx * outDim + out_channel] = fmax(0.0f, sum);
}
