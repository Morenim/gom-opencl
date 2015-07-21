package main

import (
	"fmt"
	"github.com/Morenim/gom-opencl/bitset"
	"github.com/rainliu/gocl/cl"
	"io/ioutil"
	"log"
	"unsafe"
)

func Evaluate(k int, bits bitset.BitSet) (fitness float64) {
	for i := 0; i < bits.Len()/k; i++ {
		t := 0 // number of bits set to 1
		for j := 0; j < k; j++ {
			if bits.Has(i*k + j) {
				t++
			}
		}
		if t == k {
			fitness += float64(t)
		} else {
			fitness += float64(k - t - 1)
		}
	}
	return
}

func main() {
	var status cl.CL_int

	var numPlatforms cl.CL_uint

	//---------------------------------------------------
	// Step 1: Discover and retrieve OpenCL platforms.
	//---------------------------------------------------

	status = cl.CLGetPlatformIDs(0, nil, &numPlatforms)

	platforms := make([]cl.CL_platform_id, numPlatforms)

	status = cl.CLGetPlatformIDs(numPlatforms, platforms, nil)

	if status != cl.CL_SUCCESS {
		log.Fatalf("Fatal error: could not retrieve OpenCL platform IDs.")
	}

	// Print debug info for the platforms.

	log.Printf("Debug: found %d platforms:", numPlatforms)

	getParam := func(id cl.CL_platform_id, name cl.CL_platform_info) interface{} {
		var numChars cl.CL_size_t
		var info interface{}

		status := cl.CLGetPlatformInfo(id, name, 0, nil, &numChars)
		status = cl.CLGetPlatformInfo(id, name, numChars, &info, nil)

		if status != cl.CL_SUCCESS {
			log.Fatalf("Fatal error: could not retrieve OpenCL platform info for id %d", id)
		}

		return info.(string)
	}

	for _, id := range platforms {
		log.Printf("%s %d", "PlatformID", id)
		log.Printf("\t%-11s: %s", "Name", getParam(id, cl.CL_PLATFORM_NAME))
		log.Printf("\t%-11s: %s", "Vendor", getParam(id, cl.CL_PLATFORM_VENDOR))
		log.Printf("\t%-11s: %s", "Version", getParam(id, cl.CL_PLATFORM_VERSION))
		log.Printf("\t%-11s: %s", "Profile", getParam(id, cl.CL_PLATFORM_PROFILE))
		log.Printf("\t%-11s: %s", "Extensions", getParam(id, cl.CL_PLATFORM_EXTENSIONS))
	}

	//---------------------------------------------------
	// Step 2: Discover and retrieve OpenCL devices.
	//---------------------------------------------------

	var numDevices cl.CL_uint

	status = cl.CLGetDeviceIDs(platforms[0], cl.CL_DEVICE_TYPE_ALL, 0, nil, &numDevices)

	devices := make([]cl.CL_device_id, numDevices)

	status = cl.CLGetDeviceIDs(
		platforms[0],
		cl.CL_DEVICE_TYPE_ALL,
		numDevices,
		devices,
		nil)

	if status != cl.CL_SUCCESS {
		log.Fatalf("Fatal error: could not retrieve OpenCL device IDs.")
	}

	// Debug log info.
	log.Printf("Debug: found %d devices.", numDevices)

	//---------------------------------------------------
	// Step 3: Create an OpenCL context.
	//---------------------------------------------------

	context := cl.CLCreateContext(nil, numDevices, devices, nil, nil, &status)

	if status != cl.CL_SUCCESS {
		log.Fatalf("Fatal error: could not create OpenCL context.")
	}

	defer cl.CLReleaseContext(context)

	//---------------------------------------------------
	// Step 3: Create an OpenCL command queue.
	//---------------------------------------------------

	commandQueue := cl.CLCreateCommandQueue(context, devices[0], 0, &status)

	if status != cl.CL_SUCCESS {
		log.Fatalf("Fatal error: could not create OpenCL command queue.")
	}

	defer cl.CLReleaseCommandQueue(commandQueue)

	//---------------------------------------------------
	// Step 4: Create OpenCL program and kernel.
	//---------------------------------------------------

	var kernelSource [1][]byte
	var kernelLength [1]cl.CL_size_t
	filename := "deceptive.cl"

	kernelData, err := ioutil.ReadFile(filename)

	if err != nil {
		log.Fatalf("Could not read the kernel file %s.", filename)
	}

	kernelSource[0] = kernelData
	kernelLength[0] = cl.CL_size_t(len(kernelSource[0]))

	program := cl.CLCreateProgramWithSource(context, 1, kernelSource[:], kernelLength[:], &status)

	if status != cl.CL_SUCCESS {
		log.Fatal("Fatal error: could not compile an OpenCL kernel from source.")
	}

	status = cl.CLBuildProgram(program, numDevices, devices, nil, nil, nil)

	if status != cl.CL_SUCCESS {
		log.Print("Fatal error: could not build OpenCL program.")

		var numChars cl.CL_size_t
		var info interface{}

		status = cl.CLGetProgramBuildInfo(
			program, devices[0], cl.CL_PROGRAM_BUILD_LOG,
			0, nil, &numChars)

		status = cl.CLGetProgramBuildInfo(
			program, devices[0], cl.CL_PROGRAM_BUILD_LOG,
			numChars, &info, nil)

		if status != cl.CL_SUCCESS {
			log.Fatal("Fatal error: could not retrieve OpenCL program build info.")
		}

		log.Fatalf("%s", info.(string))
	}

	kernel := cl.CLCreateKernel(program, []byte("deceptive"), &status)

	if status != cl.CL_SUCCESS {
		log.Fatal("Fatal error: could not create OpenCL kernel.")
	}

	//---------------------------------------------------
	// Step 5: Initialize the LTGA.
	//---------------------------------------------------

	populationSize := 32
	problemLength := 32

	pop := NewPopulation(populationSize, problemLength)

	freqs := Frequencies(pop)

	lt := LinkageTree(pop, freqs)

	fmt.Printf("%v\n", lt)

	//---------------------------------------------------
	// Step 6: Initialize OpenCL memory.
	//---------------------------------------------------

	var size cl.CL_uint
	popSize := cl.CL_size_t(populationSize)
	dataSize := cl.CL_size_t(unsafe.Sizeof(size)) * popSize

	populationData := make([]cl.CL_uint, populationSize)

	for i, solution := range pop.Solutions {
		var raw uint32
		raw = 0

		var j uint32
		for j = 0; j < 32; j++ {
			if solution.Bits.Has(int(j)) {
				raw |= (1 << j)
			}
		}

		populationData[i] = cl.CL_uint(raw)
	}

	offspringData := make([]cl.CL_uint, populationSize)

	populationBuffer := cl.CLCreateBuffer(
		context, cl.CL_MEM_READ_ONLY, dataSize, nil, &status)

	if status != cl.CL_SUCCESS {
		log.Fatal("Fatal error: could not allocate an OpenCL memory buffer.")
	}

	offspringBuffer := cl.CLCreateBuffer(
		context, cl.CL_MEM_WRITE_ONLY, dataSize, nil, &status)

	if status != cl.CL_SUCCESS {
		log.Fatal("Fatal error: could not allocate an OpenCL memory buffer.")
	}

	status = cl.CLEnqueueWriteBuffer(
		commandQueue, populationBuffer, cl.CL_TRUE, 0,
		dataSize, unsafe.Pointer(&populationData[0]), 0, nil, nil)

	if status != cl.CL_SUCCESS {
		log.Fatal("Fatal error: could not write data to an OpenCL memory buffer.")
	}

	status = cl.CLSetKernelArg(
		kernel, 0, cl.CL_size_t(unsafe.Sizeof(populationBuffer)),
		unsafe.Pointer(&populationBuffer))

	if status != cl.CL_SUCCESS {
		fmt.Println(status)
		log.Fatal("Fatal error: could not set arg 0 for OpenCL kernel.")
	}

	status = cl.CLSetKernelArg(
		kernel, 1, cl.CL_size_t(unsafe.Sizeof(offspringBuffer)),
		unsafe.Pointer(&offspringBuffer))

	if status != cl.CL_SUCCESS {
		log.Fatal("Fatal error: could not set arg 1 for OpenCL kernel.")
	}

	var globalWorkSize [1]cl.CL_size_t
	globalWorkSize[0] = cl.CL_size_t(populationSize)

	//---------------------------------------------------
	// Step 7: Perform GOM crossover.
	//---------------------------------------------------

	status = cl.CLEnqueueNDRangeKernel(
		commandQueue, kernel, 1, nil, globalWorkSize[:],
		nil, 0, nil, nil)

	if status != cl.CL_SUCCESS {
		log.Fatal("Fatal error: could not enqueue OpenCL kernel.")
	}

	cl.CLEnqueueReadBuffer(
		commandQueue, offspringBuffer, cl.CL_TRUE, 0,
		dataSize, unsafe.Pointer(&offspringData[0]), 0, nil, nil)

	if status != cl.CL_SUCCESS {
		log.Fatal("Fatal error: reading a buffer failed.")
	}

	for i, fitness := range offspringData {
		log.Printf("dt(%-2d) = %-2d, bits = %032b, out(%-2d) = %-2d",
			i, int(Evaluate(4, pop.Solutions[i].Bits)),
			populationData[i], i, fitness)
	}
}
