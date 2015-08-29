package main

import (
	"flag"
	"fmt"
	"github.com/Morenim/gom-opencl/bitset"
	"github.com/rainliu/gocl/cl"
	"io/ioutil"
	"log"
	"math/rand"
	"time"
	"unsafe"
)

var (
	useCPU         bool
	verbosity      int
	randomSeed     int
	populationSize int
	numGenerations int
	problemLength  int
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

type byLength [][]int

func (bl byLength) Len() int {
	return len(bl)
}

func (bl byLength) Swap(i, j int) {
	bl[i], bl[j] = bl[j], bl[i]
}

func (bl byLength) Less(i, j int) bool {
	if len(bl[i]) < len(bl[j]) {
		return true
	}
	if len(bl[i]) == len(bl[j]) {
		return bl[i][0] < bl[j][0]
	}
	return false
}

var deviceErrorMap = map[cl.CL_int]string{
	cl.CL_SUCCESS:             "cl: Success",
	cl.CL_DEVICE_NOT_FOUND:    "cl: Device Not Found",
	cl.CL_OUT_OF_HOST_MEMORY:  "cl: Out of Host Memory",
	cl.CL_OUT_OF_RESOURCES:    "cl: Out of Resources",
	cl.CL_INVALID_VALUE:       "cl: Invalid Value",
	cl.CL_INVALID_PLATFORM:    "cl: Invalid Platform",
	cl.CL_INVALID_DEVICE_TYPE: "cl: Invalid Device Type",
}

// Find the first device matching the device type from the list of platforms.
func findDevice(platforms []cl.CL_platform_id, deviceType cl.CL_device_type) (platformID cl.CL_platform_id, deviceID cl.CL_device_id) {

	// Search all platforms for the first device.
	for _, platform := range platforms {

		var numDevices cl.CL_uint

		// Get the number of matching devices for the platform.
		status := cl.CLGetDeviceIDs(platform, deviceType, 0, nil, &numDevices)

		// Check for errors, continue to next platform if no matching device was found.
		switch status {
		case cl.CL_DEVICE_NOT_FOUND:
			fallthrough
		case cl.CL_SUCCESS:
			if numDevices == 0 {
				continue
			}
		default:
			log.Printf("%s", deviceErrorMap[status])
			log.Fatalf("Fatal error: could not retrieve devices for platform %d", platform)
		}

		device := make([]cl.CL_device_id, 1)

		// Select first device matching the device type.
		status = cl.CLGetDeviceIDs(
			platform,
			cl.CL_DEVICE_TYPE_GPU,
			1,
			device,
			nil)

		if status != cl.CL_SUCCESS {
			log.Fatalf("Fatal error: could not retrieve GPU device for platform %d", platform)
		}

		platformID = platform
		deviceID = device[0]
		break
	}

	return
}

func printPlatforms(platforms []cl.CL_platform_id) {

	log.Printf("Debug: found %d platforms:", len(platforms))

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
}

func printDeviceInfo(device cl.CL_device_id) {
	var buffer interface{}

	getParam := func(name cl.CL_device_info) interface{} {
		status := cl.CLGetDeviceInfo(device, name, 128, &buffer, nil)

		if status != cl.CL_SUCCESS {
			log.Printf("%s", deviceErrorMap[status])
			log.Fatalf("Fatal error: could not retrieve work group information for kernel.")
		}

		return buffer
	}

	log.Printf("Chosen Device Info:")
	switch getParam(cl.CL_DEVICE_TYPE) {
	case cl.CL_DEVICE_TYPE_GPU:
		log.Printf("\t%-11s: %v", "Type", "GPU")
	case cl.CL_DEVICE_TYPE_CPU:
		log.Printf("\t%-11s: %v", "Type", "CPU")
	}
	log.Printf("\t%-11s: %v", "Max Compute Units", getParam(cl.CL_DEVICE_MAX_COMPUTE_UNITS))
	log.Printf("\t%-11s: %v", "Max Work Group Size", getParam(cl.CL_DEVICE_MAX_WORK_GROUP_SIZE))
	log.Printf("\t%-11s: %v", "Max Work Item Dimensions", getParam(cl.CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS))
	log.Printf("\t%-11s: %v", "Max Mem Alloc Size", getParam(cl.CL_DEVICE_MAX_MEM_ALLOC_SIZE))
	//log.Printf("\t%-11s: %v", "Max Write Image Args", getParam(cl.CL_DEVICE_MAX_READ_IMAGE_ARGS))
	//log.Printf("\t%-11s: %v", "Max Image2D Width", getParam(cl.CL_DEVICE_IMAGE2D_MAX_WIDTH))
	//log.Printf("\t%-11s: %v", "Max Image2D Height", getParam(cl.CL_DEVICE_IMAGE2D_MAX_HEIGHT))
}

func printKernelWorkGroup(kernel cl.CL_kernel, device cl.CL_device_id) {

	var buffer interface{}

	getParam := func(name cl.CL_kernel_work_group_info) interface{} {
		status := cl.CLGetKernelWorkGroupInfo(kernel, device, name, 12, &buffer, nil)

		if status != cl.CL_SUCCESS {
			log.Printf("%s", deviceErrorMap[status])
			log.Fatalf("Fatal error: could not retrieve work group information for kernel.")
		}

		return buffer
	}

	log.Printf("Kernel Work Group Information:")
	log.Printf("\t%-11s: %v", "Work Group Size", getParam(cl.CL_KERNEL_WORK_GROUP_SIZE))
	log.Printf("\t%-11s: %v", "Local Memory Size", getParam(cl.CL_KERNEL_LOCAL_MEM_SIZE))
	log.Printf("\t%-11s: %v", "Private Memory Size", getParam(cl.CL_KERNEL_PRIVATE_MEM_SIZE))
	log.Printf("\t%-11s: %v", "Preferred Work Group Size Multiple", getParam(cl.CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE))
}

func printGeneration(numGenerations int, pop *Population) {
	fmt.Printf("Generation %d\n", numGenerations)
	fmt.Println("===============")
	for i, solution := range pop.Solutions {
		solution.Fitness = Evaluate(4, solution.Bits)
		fmt.Printf("x_%-2d: %v", i, solution)
	}
	fmt.Println("===============")
	fmt.Println()
}

func flattenIntoSlice(src [][]int, dest []cl.CL_uint) {

	dest[0] = cl.CL_uint(len(src))
	i := 1
	for _, node := range src {
		j := 0
		numMasks := 0
		start := i // Reserve index for number of masks.
		i++

		for j < len(node) {
			maskIndex := uint32(node[j] >> 5) // index of mask to create
			mask := uint32(0)

			// Create a mask from all indices belonging to the block of bits.
			for j < len(node) && (uint32(node[j])>>5) == maskIndex {
				mask |= 1 << (uint32(node[j]) & 31)
				j++
			}

			// Append the mask to the flattened FOS>
			dest[i] = cl.CL_uint(maskIndex)
			dest[i+1] = cl.CL_uint(mask)
			i += 2
			numMasks++
		}

		dest[start] = cl.CL_uint(numMasks)
	}
}

func populationToSlice(pop *Population, dest []cl.CL_uint) {

	for i, solution := range pop.Solutions {
		var raw uint32
		raw = 0

		var j uint32
		for j = 0; j < 32; j++ {
			if solution.Bits.Has(int(j)) {
				raw |= (1 << j)
			}
		}

		dest[i] = cl.CL_uint(raw)
	}
}

func setKernelArg(kernel cl.CL_kernel, pos int, data *cl.CL_mem) {
	status := cl.CLSetKernelArg(
		kernel, cl.CL_uint(pos), cl.CL_size_t(unsafe.Sizeof(data)),
		unsafe.Pointer(data))

	if status != cl.CL_SUCCESS {
		log.Fatalf("Fatal error: could not set arg %d for OpenCL kernel.", pos)

	}
}

func printProgramInfo(program cl.CL_program, name cl.CL_program_info) string {

	var buffer interface{}
	var size cl.CL_size_t

	status := cl.CLGetProgramInfo(program, name, 0, nil, &size)

	status = cl.CLGetProgramInfo(program, cl.CL_PROGRAM_SOURCE, size, &buffer, nil)

	if status != cl.CL_SUCCESS {
		log.Printf("%s", cl.ERROR_CODES_STRINGS[-status])
		log.Fatal("Fatal error: could not retrieve program info.")
	}

	return fmt.Sprintf("%s", buffer.(string))
}

func parseCommandLine() {

	flag.BoolVar(&useCPU, "cpu", false, "Whether to use the CPU over the GPU.")

	flag.IntVar(&verbosity, "verbosity", 0, "Verbosity of the output.")

	flag.IntVar(&randomSeed, "random", 0, "Random seed to use. Defaults to a time-based random seed.")

	flag.IntVar(&populationSize, "size", 64, "Number of solutions in the fixed-size population.")

	flag.IntVar(&numGenerations, "generations", 10, "Maximum number of generations to perform for GOMEA.")

	flag.IntVar(&problemLength, "length", 32, "Length of the optimization problem.")

	flag.Parse()
}

func runOpenCL() {

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

	if verbosity >= 2 {
		printPlatforms(platforms)
	}

	//---------------------------------------------------
	// Step 2: Discover and retrieve OpenCL devices.
	//---------------------------------------------------

	var preferredType cl.CL_device_type

	if useCPU {
		preferredType = cl.CL_DEVICE_TYPE_CPU
	} else {
		preferredType = cl.CL_DEVICE_TYPE_GPU
	}

	_, gpuDevice := findDevice(platforms, preferredType)
	gpuDevices := make([]cl.CL_device_id, 1)
	gpuDevices[0] = gpuDevice

	if verbosity >= 1 {
		printDeviceInfo(gpuDevice)
	}

	//---------------------------------------------------
	// Step 3: Create an OpenCL context.
	//---------------------------------------------------

	context := cl.CLCreateContext(nil, 1, gpuDevices, nil, nil, &status)

	if status != cl.CL_SUCCESS {
		log.Fatalf("Fatal error: could not create OpenCL context.")
	}

	defer cl.CLReleaseContext(context)

	//---------------------------------------------------
	// Step 3: Create an OpenCL command queue.
	//---------------------------------------------------

	commandQueue := cl.CLCreateCommandQueue(context, gpuDevice, 0, &status)

	if status != cl.CL_SUCCESS {
		log.Fatalf("Fatal error: could not create OpenCL command queue.")
	}

	defer cl.CLReleaseCommandQueue(commandQueue)

	//---------------------------------------------------
	// Step 4: Create OpenCL program and kernel.
	//---------------------------------------------------

	var clSourceData [3][]byte
	var clSourceLengths [3]cl.CL_size_t
	var err error

	clSourceFiles := []string{
		"kernels/deceptive_trap.cl",
		"kernels/rng.cl",
		"kernels/gom.cl",
	}

	for i, s := range clSourceFiles {
		clSourceData[i], err = ioutil.ReadFile(s)

		if err != nil {
			log.Fatalf("Could not read the kernel source file %s.", s)
		}

		clSourceLengths[i] = cl.CL_size_t(len(clSourceData[i]))
	}

	program := cl.CLCreateProgramWithSource(context, 3, clSourceData[:], clSourceLengths[:], &status)

	if status != cl.CL_SUCCESS {
		log.Printf("%s", deviceErrorMap[status])
		log.Fatal("Fatal error: could not compile an OpenCL kernel from source.")
	}

	status = cl.CLBuildProgram(program, 1, gpuDevices, nil, nil, nil)

	if status != cl.CL_SUCCESS {
		log.Print("Fatal error: could not build OpenCL program.")

		var numChars cl.CL_size_t
		var info interface{}

		status = cl.CLGetProgramBuildInfo(
			program, gpuDevice, cl.CL_PROGRAM_BUILD_LOG,
			0, nil, &numChars)

		status = cl.CLGetProgramBuildInfo(
			program, gpuDevice, cl.CL_PROGRAM_BUILD_LOG,
			numChars, &info, nil)

		if status != cl.CL_SUCCESS {
			log.Fatal("Fatal error: could not retrieve OpenCL program build info.")
		}

		log.Fatalf("%s", info.(string))
	}

	kernel := cl.CLCreateKernel(program, []byte("gom"), &status)

	if status != cl.CL_SUCCESS {
		log.Fatal("Fatal error: could not create OpenCL kernel.")
	}

	//---------------------------------------------------
	// Step 6: Initialize OpenCL memory.
	//---------------------------------------------------

	if verbosity >= 1 {
		printKernelWorkGroup(kernel, gpuDevice)
	}

	//---------------------------------------------------
	// Step 7: Initialize OpenCL memory.
	//---------------------------------------------------

	var size cl.CL_uint
	popSize := cl.CL_size_t(populationSize)
	length := cl.CL_size_t(problemLength)

	pop := NewPopulation(populationSize, problemLength)

	dataSize := cl.CL_size_t(unsafe.Sizeof(size)) * popSize

	populationData := make([]cl.CL_uint, pop.Size())

	offspringData := make([]cl.CL_uint, pop.Size())

	populationBuffer := cl.CLCreateBuffer(
		context, cl.CL_MEM_READ_ONLY, dataSize, nil, &status)

	if status != cl.CL_SUCCESS {
		log.Fatal("Fatal error: could not allocate an OpenCL memory buffer.")
	}

	// Maximum bound on the number of elements in the LT + node sizes.
	boundSum := (length*length+3*length-2)/2 + (2*length - 1) + 1
	ltSize := cl.CL_size_t(unsafe.Sizeof(length)) * boundSum

	ltData := make([]cl.CL_uint, boundSum)

	ltBuffer := cl.CLCreateBuffer(
		context, cl.CL_MEM_READ_ONLY, ltSize, nil, &status)

	if status != cl.CL_SUCCESS {
		log.Fatal("Fatal error: could not allocate an OpenCL memory buffer.")
	}

	offspringBuffer := cl.CLCreateBuffer(
		context, cl.CL_MEM_WRITE_ONLY, dataSize, nil, &status)

	if status != cl.CL_SUCCESS {
		log.Fatal("Fatal error: could not allocate an OpenCL memory buffer.")
	}

	//---------------------------------------------------
	// Step 8: Perform GOMEA.
	//---------------------------------------------------

	if randomSeed == 0 {
		rand.Seed(time.Now().Unix())
	} else {
		rand.Seed(int64(randomSeed))
	}

	done := false

	generationsPassed := 0

	for !done {

		if (verbosity == 2 && generationsPassed == numGenerations-1) || (verbosity == 3) {
			printGeneration(generationsPassed, pop)
		}

		// Build the linkage tree.
		freqs := Frequencies(pop)

		lt := LinkageTree(pop, freqs)

		//log.Printf("%v", lt)

		// Store a flattened version of the linkage tree in memory.
		flattenIntoSlice(lt, ltData)

		//log.Printf("%v", ltData)

		// Upload FOS.
		status = cl.CLEnqueueWriteBuffer(
			commandQueue, ltBuffer, cl.CL_TRUE, 0,
			ltSize, unsafe.Pointer(&ltData[0]), 0, nil, nil)

		if status != cl.CL_SUCCESS {
			log.Fatal("Fatal error: could not write data to an OpenCL memory buffer.")
		}

		populationToSlice(pop, populationData)

		status = cl.CLEnqueueWriteBuffer(
			commandQueue, populationBuffer, cl.CL_TRUE, 0,
			dataSize, unsafe.Pointer(&populationData[0]), 0, nil, nil)

		if status != cl.CL_SUCCESS {
			log.Fatal("Fatal error: could not write data to an OpenCL memory buffer.")
		}

		// Set the GOM kernel arguments.
		setKernelArg(kernel, 0, &populationBuffer)

		clSize := cl.CL_uint(populationSize)
		status := cl.CLSetKernelArg(
			kernel, 1, cl.CL_size_t(unsafe.Sizeof(clSize)),
			unsafe.Pointer(&popSize))

		if status != cl.CL_SUCCESS {
			log.Printf("%v", cl.ERROR_CODES_STRINGS[-status])
			log.Fatalf("Fatal error: could not set arg %d for OpenCL kernel.", 1)
		}

		setKernelArg(kernel, 2, &ltBuffer)
		setKernelArg(kernel, 3, &offspringBuffer)

		var globalWorkSize [1]cl.CL_size_t
		globalWorkSize[0] = cl.CL_size_t(pop.Size())

		// Perform GOM crossover.
		status = cl.CLEnqueueNDRangeKernel(
			commandQueue, kernel, 1, nil, globalWorkSize[:],
			nil, 0, nil, nil)

		if status != cl.CL_SUCCESS {
			log.Fatal("Fatal error: could not enqueue OpenCL kernel.")
		}

		status = cl.CLFinish(commandQueue)

		if status != cl.CL_SUCCESS {
			log.Printf("%s", cl.ERROR_CODES_STRINGS[-status])
			log.Fatal("Fatal error: could not finish command queue.")
		}

		// Retrieve the offspring population from the compute device.
		cl.CLEnqueueReadBuffer(
			commandQueue, offspringBuffer, cl.CL_TRUE, 0,
			dataSize, unsafe.Pointer(&offspringData[0]), 0, nil, nil)

		if status != cl.CL_SUCCESS {
			log.Fatal("Fatal error: reading a buffer failed.")
		}

		for i, offspring := range offspringData {
			pop.Solutions[i].Bits, _ = bitset.FromString(fmt.Sprintf("%032b", uint(offspring)))
			pop.Solutions[i].Fitness = Evaluate(4, pop.Solutions[i].Bits)
		}

		generationsPassed++

		// TODO: Termination Criterion
		if generationsPassed == numGenerations {
			done = true
		}
	}
}

func main() {

	parseCommandLine()

	runOpenCL()
}
