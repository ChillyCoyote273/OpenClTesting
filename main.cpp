#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstdlib>
#include <time.h>


std::string get_kernel_code(std::string file_name) {
	std::ifstream t(file_name);
	std::stringstream buffer;
	buffer << t.rdbuf();
	return buffer.str();
}

int main(int argc, char** argv) {
	std::cout << "\n";

	std::vector<cl::Platform> allPlatforms;
	cl::Platform::get(&allPlatforms);
	if (allPlatforms.size() == 0) {
		std::cout << "No OpenCL platforms found.\n";
		exit(1);
	}

	cl::Platform defaultPlatform = allPlatforms[0];
	std::cout << "Using platform: " << defaultPlatform.getInfo<CL_PLATFORM_NAME>() << "\n";

	std::vector<cl::Device> allDevices;
	defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);
	if (allDevices.size() == 0) {
		std::cout << "No OpenCL devices found.\n";
		exit(1);
	}

	cl::Device defaultDevice = allDevices[0];
	std::cout << "Using device: " << defaultDevice.getInfo<CL_DEVICE_NAME>() << "\n";
	std::cout << "Max work group size: " << defaultDevice.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << "\n";
	std::cout << defaultDevice.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
	std::cout << defaultDevice.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "\n";
	auto work_item_sizes = defaultDevice.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
	std::cout << "Max work item sizes: ";
	for (auto item_size: work_item_sizes) {
		std::cout << item_size << " ";
	}
	std::cout << "\n";

	cl::Context context({defaultDevice});

	cl::Program::Sources sources;
	std::string kernel_code = get_kernel_code("kernel.cl");
	sources.push_back({ kernel_code.c_str(), kernel_code.length() });

	int SIZE = 100000;
	int* A_h = (int*)malloc(sizeof(int) * SIZE);
	int* B_h = (int*)malloc(sizeof(int) * SIZE);
	int* C_h = (int*)malloc(sizeof(int) * SIZE);

	srand(time(0));
	for (int i = 0; i < SIZE; i++) {
		A_h[i] = rand() / 2;
		B_h[i] = rand() / 2;
	}

	cl::Buffer A_d(context, CL_MEM_READ_ONLY, sizeof(int) * SIZE);
	cl::Buffer B_d(context, CL_MEM_READ_ONLY, sizeof(int) * SIZE);
	cl::Buffer C_d(context, CL_MEM_WRITE_ONLY, sizeof(int) * SIZE);

	cl::CommandQueue queue(context, defaultDevice, CL_QUEUE_PROFILING_ENABLE);
	queue.enqueueWriteBuffer(A_d, CL_TRUE, 0, sizeof(int) * SIZE, A_h);
	queue.enqueueWriteBuffer(B_d, CL_TRUE, 0, sizeof(int) * SIZE, B_h);
	
	cl::Program program(context, sources);
	if (program.build({ defaultDevice }) != CL_SUCCESS) {
		std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(defaultDevice) << "\n";
		exit(1);
	}
	std::cout << "Kernel built.\n";

	cl::Kernel simpleAdd(program, "simple_add");
	simpleAdd.setArg(0, A_d);
	simpleAdd.setArg(1, B_d);
	simpleAdd.setArg(2, C_d);

	size_t time_queued;
	size_t time_start;
	size_t time_end;
	cl::Event event;

	cl::NDRange global(SIZE);
	std::cout << "Starting kernel.\n";
	queue.finish();
	// auto start = std::chrono::high_resolution_clock::now();
	queue.enqueueNDRangeKernel(simpleAdd, cl::NullRange, global, cl::NullRange, NULL, &event);
	// auto stop = std::chrono::high_resolution_clock::now();
	// auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	// std::cout << "Kernel executed in " << duration.count() / 1000 << " milliseconds and " << duration.count() % 1000 << " microseconds.\n";
	cl::WaitForEvents({event});
	queue.finish();
	event.getProfilingInfo<size_t>(CL_PROFILING_COMMAND_QUEUED, &time_queued);
	event.getProfilingInfo<size_t>(CL_PROFILING_COMMAND_START, &time_start);
	event.getProfilingInfo<size_t>(CL_PROFILING_COMMAND_END, &time_end);

	uint64_t nanoseconds = time_start - time_queued;
	std::cout << "Startup latency was " << nanoseconds / 1000000 << " milliseconds, " << nanoseconds / 1000 % 1000 << " microseconds, and " << nanoseconds % 1000 << " nanoseconds.\n";
	nanoseconds = time_end - time_start;
	std::cout << "Kernel took " << nanoseconds / 1000000 << " milliseconds, " << nanoseconds / 1000 % 1000 << " microseconds, and " << nanoseconds % 1000 << " nanoseconds.\n";

	queue.enqueueReadBuffer(C_d, CL_TRUE, 0, sizeof(int) * SIZE, C_h);

	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < SIZE; i++) {
		if (A_h[i] + B_h[i] != C_h[i]) {
			std::cout << A_h[i] << " + " << B_h[i] << " = " << C_h[i] << "\n";
		}
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "CPU executed in " << duration.count() / 1000 << " milliseconds and " << duration.count() % 1000 << " microseconds.\n";

	std::cout << "\n";
	return 0;
}