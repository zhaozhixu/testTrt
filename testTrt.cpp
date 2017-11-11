#include <cassert>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <dirent.h>
#include <unistd.h>

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "common.h"
#include "tensorUtil.h"

static Logger gLogger;
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

int INPUT_C = 3;
int INPUT_H = 4;
int INPUT_W = 4;
int OUTPUT_SIZE = 18;

// Our weight files are in a very simple space delimited format.
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::map<std::string, Weights> weightMap;
	std::ifstream input(file);
	assert(input.is_open() && "Unable to load weight file.");
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");
    while(count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t type, size;
        std::string name;
        input >> name >> std::dec >> type >> size;
        wt.type = static_cast<DataType>(type);
        if (wt.type == DataType::kFLOAT)
        {
            uint32_t *val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size)); // wrong sizeof oprand
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> val[x];

            }
            wt.values = val;
        } else if (wt.type == DataType::kHALF)
        {
            uint16_t *val = reinterpret_cast<uint16_t*>(malloc(sizeof(val) * size)); // wrong sizeof oprand
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> val[x];
            }
            wt.values = val;
        }
        wt.count = size;
        weightMap[name] = wt;
    }
    return weightMap;
}

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"samples/testTrt/data/"};
    return locateFile(input, dirs);
}

// Creat the Engine using only the API and not any parser.
ICudaEngine *
createEngine(unsigned int maxBatchSize, IBuilder *builder, DataType dt)
{
	INetworkDefinition* network = builder->createNetwork();

	auto data = network->addInput("input", dt, DimsCHW{ 3, 4, 4});
	assert(data != nullptr);

    std::map<std::string, Weights> weightMap = loadWeights(locateFile("testtrt.wts")); // ?
    Weights omit;
    omit.count = 0;
    omit.values = NULL;
	auto conv1 = network->addConvolution(*data, 2, DimsHW{2, 2},
										 weightMap["conv1filter"],
										 omit);
	assert(conv1 != nullptr);
	// conv1->setStride(DimsHW{2, 2});
    // auto relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    // assert(relu1 != nullptr);

	conv1->getOutput(0)->setName("output");
	network->markOutput(*conv1->getOutput(0));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 20);

	auto engine = builder->buildCudaEngine(*network);
	// we don't need the network any more
	network->destroy();

	// Once we have built the cuda engine, we can release all of our held memory.
	for (auto &mem : weightMap)
    {
        free((void*)(mem.second.values));
    }
	return engine;
}

void APIToModel(unsigned int maxBatchSize, // batch size - NB must be at least as large as the batch we want to run with)
		     IHostMemory **modelStream)
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	// create the model to populate the network, then set the outputs and create an engine
	ICudaEngine* engine = createEngine(maxBatchSize, builder, DataType::kFLOAT);

	assert(engine != nullptr);
	// serialize the engine, then close everything down
	(*modelStream) = engine->serialize();
	engine->destroy();
	builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float *output, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly 1 inputs and 1 outputs.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex0 = engine.getBindingIndex("input"),
		outputIndex0 = engine.getBindingIndex("output");

	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex0], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex0], batchSize * OUTPUT_SIZE * sizeof(float)));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex0], input, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);

    int dims[] = {2, 3, 3};
    Tensor notsliced = createTensor(buffers[outputIndex0], 3, dims);
    Tensor *sliced;
    sliceTensor(notsliced, sliced, 2, 1, 2);


	CHECK(cudaMemcpyAsync(output, buffers[outputIndex0], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex0]));
	CHECK(cudaFree(buffers[outputIndex0]));
}

int main(int argc, char** argv)
{
	IHostMemory *gieModelStream{ nullptr };
	// batch size
	const int N = 1;

    APIToModel(1, &gieModelStream);
	// float* data = new float[N*INPUT_C*INPUT_H*INPUT_W];
    float data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

	// deserialize the engine
	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), nullptr);

	IExecutionContext *context = engine->createExecutionContext();


	// host memory for outputs
	float* output = new float[N * OUTPUT_SIZE];

	// run inference
	doInference(*context, data, output, N);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("%.2f ", output[i]);
    }
    printf("\n");

	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();

	// delete[] data;
	delete[] output;
	return 0;
}
