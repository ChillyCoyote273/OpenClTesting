#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>
#include <iostream>
#include <vector>


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

	return 0;
}