#define _CRT_SECURE_NO_WARNINGS
#include "cnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <CL/cl.h>

// Error Check
#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

#define BATCH_SIZE 256

// 입력 차원 크기
const int INPUT_DIM[] = {
	3, 64,
	64,

	64,128,
	128,

	128, 256, 256,
	256,

	256, 512, 512,
	512,

	512, 512, 512,
	512,

	512,
	512,
	512
};

// 출력 차원 크기
const int OUTPUT_DIM[] = {
	64, 64,
	64,

	128, 128,
	128,

	256, 256, 256,
	256,

	512, 512, 512,
	512,

	512, 512, 512,
	512,

	512,
	512,
	10
};

// 특징맵 크기
const int NBYN[] = {
	32, 32,
	16,

	16, 16,
	8,

	8, 8, 8,
	4,

	4, 4, 4,
	2,

	2, 2, 2,
	1,

	1,
	1,
	1
};

cl_int err;
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue transfer_queue;
cl_command_queue compute_queue;
cl_program program;
cl_kernel convolution_layer_large;
cl_kernel convolution_layer_small;
cl_kernel max_pooling_layer;
cl_kernel fc_layer;
char* kernel_source;
size_t kernel_source_size;

cl_mem w_buf[21];
cl_mem b_buf[21];
cl_mem layer_buf[2];

// 파일에서 소스코드 불러오기 함수
char* get_source_code(const char* file_name, size_t* len) {

	FILE* file = fopen(file_name, "rb");

	if (file == NULL) {
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}

	fseek(file, 0, SEEK_END);
	size_t length = (size_t)ftell(file);
	rewind(file);

	char* source_code = (char*)malloc(length + 1);
	fread(source_code, length, 1, file);
	source_code[length] = '\0';
	fclose(file);
	*len = length;

	return source_code;
}

// 빌드 에러 처리 함수
void build_error(cl_program program, cl_device_id device, cl_int err) {
	if (err == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		char* log;

		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		CHECK_ERROR(err);

		log = (char*)malloc(log_size + 1);
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		CHECK_ERROR(err);

		log[log_size] = '\0';
		printf("Compiler error:\n%s\n", log);
		free(log);
		exit(0);
	}
}

// 강도 -> 확률 변환
static void softmax(float* input, int N) {
	int i;
	float max = input[0];
	for (i = 1; i < N; i++) {
		if (max < input[i]) max = input[i];
	}
	float sum = 0;
	for (i = 0; i < N; i++) {
		sum += exp(input[i] - max);
	}
	for (i = 0; i < N; i++) {
		input[i] = exp(input[i] - max) / (sum + 1e-7);
	}
}

// 최대값 인덱스 반환
static int find_max(float* input, int classNum) {
	int i;
	int maxIndex = 0;
	float max = 0;
	for (i = 0; i < classNum; i++) {
		if (max < input[i]) {
			max = input[i];
			maxIndex = i;
		}
	}
	return maxIndex;
}

// cnn 초기설정
void cnn_init() {
	// 플랫폼 id 얻기
	err = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR(err);

	// 디바이스 id 얻기
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK_ERROR(err);

	// 컨텍스트 생성
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	// 커맨드 큐 생성
	cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0 };
	transfer_queue = clCreateCommandQueueWithProperties(context, device, props, &err);
	CHECK_ERROR(err);
	compute_queue = clCreateCommandQueueWithProperties(context, device, props, &err);
	CHECK_ERROR(err);

	// 커널 내용 불러오기
	kernel_source = get_source_code("kernel.cl", &kernel_source_size);

	// 프로그램 생성
	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);

	// 프로그램 빌드
	err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
	build_error(program, device, err);
	CHECK_ERROR(err);

	// 커널 생성
	convolution_layer_large = clCreateKernel(program, "convolution_large", &err);
	CHECK_ERROR(err);
	convolution_layer_small = clCreateKernel(program, "convolution_small", &err);
	CHECK_ERROR(err);
	max_pooling_layer = clCreateKernel(program, "max_pooling", &err);
	CHECK_ERROR(err);
	fc_layer = clCreateKernel(program, "fc_layer", &err);
	CHECK_ERROR(err);
}

// 버퍼 메모리 정리
void cleanup_buffer_resources() {
	clReleaseKernel(convolution_layer_large);
	clReleaseKernel(convolution_layer_small);
	clReleaseKernel(max_pooling_layer);
	clReleaseKernel(fc_layer);
	clReleaseProgram(program);

	for (int i = 0; i < 2; ++i) {
		clReleaseMemObject(layer_buf[i]);
	}

	for (int i = 0; i < 21; ++i) {
		if (i == 2 || i == 5 || i == 9 || i == 13 || i == 17) i++;
		clReleaseMemObject(w_buf[i]);
		clReleaseMemObject(b_buf[i]);
	}

	clReleaseCommandQueue(transfer_queue);
	clReleaseCommandQueue(compute_queue);
	clReleaseContext(context);
	free(kernel_source);
	clReleaseDevice(device);
}

// CNN 메인 함수
void cnn(float* images, float* network, int* labels, float* confidences, int num_of_image) {
	float* w[21];   // 가중치 포인터 배열
	float* b[21];   // 편향 포인터 배열
	int offset = 0; // network 배열 내 현재 위치

	// Convolution layer parameters 설정 (0-16 레이어)
	for (int i = 0; i < 17; ++i) {
		if (i == 2 || i == 5 || i == 9 || i == 13) i++; // Pooling layer (파라미터 없음)
		w[i] = network + offset;
		offset += 3 * 3 * INPUT_DIM[i] * OUTPUT_DIM[i];
		b[i] = network + offset;
		offset += OUTPUT_DIM[i];
	}

	// Fully connected layer parameters 설정 (18-20 레이어)
	for (int i = 18; i < 21; ++i) {
		w[i] = network + offset;
		offset += INPUT_DIM[i] * OUTPUT_DIM[i];
		b[i] = network + offset;
		offset += OUTPUT_DIM[i];
	}

	cnn_init();  // OpenCL 초기화 (플랫폼, 디바이스, 컨텍스트, 커맨드 큐, 커널 등 초기 생성)

	// 최대 버퍼 크기 계산 (모든 레이어 중 가장 큰 출력 크기)
	size_t max_buffer_size = 0;
	for (int i = 0; i < 21; i++) {
		size_t current_size = BATCH_SIZE * OUTPUT_DIM[i] * NBYN[i] * NBYN[i];
		if (max_buffer_size < current_size) max_buffer_size = current_size;
	}

	// 2개의 메인 버퍼 생성 (입력, 출력용 ping-pong 버퍼)
	/*
	* layer_buf[0]     layer_buf[1]
	* [Input]    ->   [Output]   (Layer 1)
	*                   ↓
	* [Input]    <-   [Output]    (Layer 2)
	* ↓
	* [Input]    ->   [Output]   (Layer 3)
	*                   ↓
	* [Input]    <-   [Output]    (Layer 4)
	*/
	for (int i = 0; i < 2; i++) {
		layer_buf[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * max_buffer_size, NULL, &err);
		CHECK_ERROR(err);
	}

	// 가중치와 편향을 위한 OpenCL 버퍼 생성
	for (int i = 0; i < 21; ++i) {
		if (i == 2 || i == 5 || i == 9 || i == 13 || i == 17) i++;  // Pooling layer 건너뛰기
		if (i < 17) {  // Convolution layer
			w_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 3 * 3 * INPUT_DIM[i] * OUTPUT_DIM[i], w[i], &err);
		}
		else {         // Fully connected layer
			w_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * INPUT_DIM[i] * OUTPUT_DIM[i], w[i], &err);
		}
		CHECK_ERROR(err);

		b_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * OUTPUT_DIM[i], b[i], &err);
		CHECK_ERROR(err);
	}

	float* batch_results = (float*)malloc(sizeof(float) * BATCH_SIZE * 10);  // 배치 결과 저장 버퍼

	/*
	* Queue1 (Host → Device)           Queue2 (Compute)              Queue1 (Device → Host)
	*    ↓                                 ↓                              ↓
	* [Write Event] ----→ [Compute Event 1] --→ [Compute Event 2] --→ [Read Event]
	*                             ↓                      ↓
	*						 [Layer 1 완료]         [Layer 2 완료]
	*/

	cl_event write_event = NULL;   // 입력 전송
	cl_event compute_event = NULL; // 연산 실행
	cl_event read_event = NULL;    // 출력 결과읽기

	time_t start, end;
	start = clock();

	// 배치 단위로 이미지 처리
	for (int batch_start = 0; batch_start < num_of_image; batch_start += BATCH_SIZE) {
		int current_batch_size = (batch_start + BATCH_SIZE <= num_of_image) ? BATCH_SIZE : (num_of_image - batch_start);

		err = clEnqueueWriteBuffer(transfer_queue, layer_buf[0], CL_FALSE, 0,
			sizeof(float) * current_batch_size * 32 * 32 * 3,
			images + batch_start * (32 * 32 * 3),
			0, NULL, &write_event);
		CHECK_ERROR(err);

		cl_event prev_event = write_event;

		for (int j = 0; j < 21; ++j) {
			int in_idx = (j % 2);
			int out_idx = ((j + 1) % 2);

			if (j == 2 || j == 5 || j == 9 || j == 13 || j == 17) { // Pooling layer
				size_t global_size[2] = { NBYN[j], OUTPUT_DIM[j] * current_batch_size * NBYN[j] };
				size_t local_size[2] = { 1, 128 };

				err = clSetKernelArg(max_pooling_layer, 0, sizeof(cl_mem), &layer_buf[in_idx]);
				err |= clSetKernelArg(max_pooling_layer, 1, sizeof(cl_mem), &layer_buf[out_idx]);
				err |= clSetKernelArg(max_pooling_layer, 2, sizeof(int), &(NBYN[j]));
				err |= clSetKernelArg(max_pooling_layer, 3, sizeof(int), &(OUTPUT_DIM[j]));
				CHECK_ERROR(err);

				err = clEnqueueNDRangeKernel(compute_queue, max_pooling_layer, 2, NULL,
					global_size, local_size,
					1, &prev_event, &compute_event);
				CHECK_ERROR(err);
			}
			else if (j <= 16) { // convolution layer
				size_t global_size[2] = { NBYN[j], OUTPUT_DIM[j] * current_batch_size * NBYN[j] };
				size_t local_size[2];
				cl_kernel conv_kernel;
				if (j <= 1) { // 32x32
					conv_kernel = convolution_layer_large;
					local_size[0] = 32;
					local_size[1] = 8;
				}
				else if (j <= 4) { // 16x16
					conv_kernel = convolution_layer_large;
					local_size[0] = 16;
					local_size[1] = 16;
				}
				else if (j <= 8) { // 8x8
					conv_kernel = convolution_layer_large;
					local_size[0] = 8;
					local_size[1] = 8;
				}
				else {  // 4x4, 2x2
					conv_kernel = convolution_layer_small;
					local_size[0] = 2;
					local_size[1] = 64;
				}

				err = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &layer_buf[in_idx]);
				err |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &layer_buf[out_idx]);
				err |= clSetKernelArg(conv_kernel, 2, sizeof(cl_mem), &w_buf[j]);
				err |= clSetKernelArg(conv_kernel, 3, sizeof(cl_mem), &b_buf[j]);
				err |= clSetKernelArg(conv_kernel, 4, sizeof(int), &(INPUT_DIM[j]));
				err |= clSetKernelArg(conv_kernel, 5, sizeof(int), &(OUTPUT_DIM[j]));
				err |= clSetKernelArg(conv_kernel, 6, sizeof(int), &(NBYN[j]));
				CHECK_ERROR(err);

				err = clEnqueueNDRangeKernel(compute_queue, conv_kernel,
					2, NULL, global_size, local_size,
					1, &prev_event, &compute_event);
				CHECK_ERROR(err);
			}
			else { // Fully connected layer
				size_t global_size[2] = { OUTPUT_DIM[j], current_batch_size };

				err = clSetKernelArg(fc_layer, 0, sizeof(cl_mem), &layer_buf[in_idx]);
				err |= clSetKernelArg(fc_layer, 1, sizeof(cl_mem), &layer_buf[out_idx]);
				err |= clSetKernelArg(fc_layer, 2, sizeof(cl_mem), &w_buf[j]);
				err |= clSetKernelArg(fc_layer, 3, sizeof(cl_mem), &b_buf[j]);
				err |= clSetKernelArg(fc_layer, 4, sizeof(int), &(INPUT_DIM[j]));
				err |= clSetKernelArg(fc_layer, 5, sizeof(int), &(OUTPUT_DIM[j]));
				CHECK_ERROR(err);

				err = clEnqueueNDRangeKernel(compute_queue, fc_layer, 2, NULL,
					global_size, NULL,
					1, &prev_event, &compute_event);
				CHECK_ERROR(err);
			}

			if (prev_event != write_event) {
				clReleaseEvent(prev_event);
			}
			prev_event = compute_event;
		}

		err = clEnqueueReadBuffer(transfer_queue, layer_buf[1], CL_FALSE, 0,
			sizeof(float) * current_batch_size * 10, batch_results,
			1, &compute_event, &read_event);
		CHECK_ERROR(err);

		clWaitForEvents(1, &read_event);

		for (int i = 0; i < current_batch_size; i++) {
			float* current_result = batch_results + (i * 10);
			softmax(current_result, 10);
			labels[batch_start + i] = find_max(current_result, 10);
			confidences[batch_start + i] = current_result[labels[batch_start + i]];
		}

		clReleaseEvent(write_event);
		clReleaseEvent(compute_event);
		clReleaseEvent(read_event);
	}

	end = clock();
	printf("Elapsed time: %.2f sec\n", (double)(end - start) / CLK_TCK);
	free(batch_results);
	cleanup_buffer_resources();
}
