#include  "Metacommand.h"
#include <dxgi1_6.h>
#include <comdef.h>
#include "d3dx12.h"
#include "utils.h"

#include "shared.h"
#include "shaders.h"
#include <chrono>
#include <thread>

const bool useFp16 = false;
const bool useMetacommands = true;

#define checkResult(ans) { _checkResult((ans), __FILE__, __LINE__); }
inline void _checkResult(HRESULT hr, const char* file, int line) {
    if (hr != S_OK) {
        _com_error err(hr);
        LPCTSTR errMsg = err.ErrorMessage();

        printf("Error: %s in file %s, line no: %d\n", errMsg, file, line);
        __debugbreak();
        exit(hr);
    }
}

struct D3D12Alloc 
{
    ID3D12Resource* pResource;
    uint32_t offset;    // offset within pResource (for suballocated resources)
    uint64_t gpuVA;
    D3D12_GPU_DESCRIPTOR_HANDLE descHandle;
};

// handle DX stuff
class D3d12Wrapper
{
private:
    ID3D12Device5 *m_pDevice;
    ID3D12CommandAllocator* m_pCA;
    ID3D12CommandQueue* m_pCQ;
    ID3D12GraphicsCommandList4* m_pCL;
    ID3D12Fence *m_pFence;
    UINT64 m_fenceVal = 0ull;

    ID3D12QueryHeap* m_pQueryHeap;
    D3D12Alloc m_queryResult;

    ID3D12DescriptorHeap *m_pDescHeap;
    static constexpr int MAX_DESCS = 32;
    int nextFreeDescHeapSlot;

public:
    void init(int gpuIndex)
    {
        IDXGIFactory4 *pFactory = nullptr;
        IDXGIAdapter *pAdapter = nullptr;
        checkResult(CreateDXGIFactory2(0, IID_PPV_ARGS(&pFactory)));
        checkResult(pFactory->EnumAdapters(gpuIndex, &pAdapter));
        pFactory->Release();

        checkResult(D3D12CreateDevice(pAdapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_pDevice)));
        pAdapter->Release();

        D3D12_COMMAND_QUEUE_DESC cqDesc = {};
        cqDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

        checkResult(m_pDevice->CreateCommandQueue(&cqDesc, IID_PPV_ARGS(&m_pCQ)));
        checkResult(m_pDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_pCA)));
        checkResult(m_pDevice->CreateCommandList(1, D3D12_COMMAND_LIST_TYPE_DIRECT, m_pCA, nullptr, IID_PPV_ARGS(&m_pCL)));

        D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        heapDesc.NumDescriptors = MAX_DESCS;
        checkResult(m_pDevice->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_pDescHeap)));
        nextFreeDescHeapSlot = 0;
        m_pCL->SetDescriptorHeaps(1, &m_pDescHeap);
        m_fenceVal = 0ull;
        checkResult(m_pDevice->CreateFence(m_fenceVal, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_pFence)));

        D3D12_QUERY_HEAP_DESC queryHeapDesc = {};
        queryHeapDesc.Count = 2;
        queryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        queryHeapDesc.NodeMask = 1;
        checkResult(m_pDevice->CreateQueryHeap(&queryHeapDesc, IID_PPV_ARGS(& m_pQueryHeap)));

        createAlloc(sizeof(uint64_t) * 2, D3D12_HEAP_TYPE_READBACK, &m_queryResult);
    }

    void createAlloc(size_t size, D3D12_HEAP_TYPE type, D3D12Alloc* pAlloc) 
    {
        // some alignment
        size_t factor = ((size - 1)/4) + 1;
        size = factor * 4;

        D3D12_HEAP_PROPERTIES heapDesc = {};
        heapDesc.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heapDesc.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heapDesc.CreationNodeMask = 1;
        heapDesc.VisibleNodeMask = 1;

        if (type == D3D12_HEAP_TYPE_CUSTOM) {
            // Use custom heap type to allow GPU writing to system memory directly
            heapDesc.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
            heapDesc.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
        }

        heapDesc.Type = type;

        D3D12_RESOURCE_DESC bufferDesc = {};
        bufferDesc.MipLevels = 1;
        bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
        bufferDesc.Height = 1;
        if (type == D3D12_HEAP_TYPE_DEFAULT || type == D3D12_HEAP_TYPE_CUSTOM)
            bufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        bufferDesc.DepthOrArraySize = 1;
        bufferDesc.SampleDesc.Count = 1;
        bufferDesc.SampleDesc.Quality = 0;
        bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        D3D12_RESOURCE_STATES resourceState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        if (type == D3D12_HEAP_TYPE_UPLOAD)
            resourceState = D3D12_RESOURCE_STATE_GENERIC_READ;
        else if (type == D3D12_HEAP_TYPE_READBACK)
            resourceState = D3D12_RESOURCE_STATE_COPY_DEST;

        bufferDesc.Width = size;
        checkResult(m_pDevice->CreateCommittedResource(
            &heapDesc, D3D12_HEAP_FLAG_NONE, &bufferDesc, resourceState, nullptr,
            IID_PPV_ARGS(&pAlloc->pResource)));

        pAlloc->offset = 0;
        pAlloc->gpuVA = pAlloc->pResource->GetGPUVirtualAddress();

        // Create desc heap entry for UAV resources.
        if (resourceState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            int slot = nextFreeDescHeapSlot++;

            int handleIncrementSize = m_pDevice->GetDescriptorHandleIncrementSize(
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

            CD3DX12_CPU_DESCRIPTOR_HANDLE cpuDescHandle(m_pDescHeap->GetCPUDescriptorHandleForHeapStart(), slot, handleIncrementSize);

            CD3DX12_GPU_DESCRIPTOR_HANDLE gpuDescHandle(m_pDescHeap->GetGPUDescriptorHandleForHeapStart(), slot, handleIncrementSize);

            D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
            uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            uavDesc.Buffer.FirstElement = 0;
            uavDesc.Format = useFp16 ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R32G32B32A32_FLOAT;
            uavDesc.Buffer.NumElements = useFp16 ? size / 8 : size / 16;

            m_pDevice->CreateUnorderedAccessView(pAlloc->pResource, nullptr, &uavDesc,
                cpuDescHandle);

            pAlloc->descHandle = gpuDescHandle;
        }
    }
    void flushAndWait() 
    {
        m_pCL->Close();
        m_pCQ->ExecuteCommandLists(1, (ID3D12CommandList**)&m_pCL);
        m_pCQ->Signal(m_pFence, ++m_fenceVal);

        // Wait for commands to finish on GPU.
        // (spinloop has lowest latency, we can try event based signal if CPU
        // overhead becomes a bottleneck).
        while (m_pFence->GetCompletedValue() != m_fenceVal) ;

        m_pCA->Reset();
        m_pCL->Reset(m_pCA, NULL);
        m_pCL->SetDescriptorHeaps(1, &m_pDescHeap);
    }

    ID3D12Device5* getDevice() { return m_pDevice; }
    ID3D12GraphicsCommandList4* getCL() { return m_pCL; }

    void beginTimer()
    {
        m_pCL->EndQuery(m_pQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
    }

    void endTimer()
    {
        m_pCL->EndQuery(m_pQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
        m_pCL->ResolveQueryData(m_pQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, m_queryResult.pResource, 0);
    }

    double getTimeInSeconds()
    {
        uint64_t freq;
        m_pCQ->GetTimestampFrequency(&freq);

        char* pCpuPointer;
        checkResult(m_queryResult.pResource->Map(0, NULL, reinterpret_cast<void**>(&pCpuPointer)));
        uint64_t* TS = (UINT64*)(pCpuPointer);
        double retVal = double((TS[1] - TS[0]) / double(freq));
        m_queryResult.pResource->Unmap(0, nullptr);

        return retVal;
    }

    void uploadData(D3D12Alloc *pAlloc, const void *pData, size_t size)
    {
        // create a staging alloc
        D3D12Alloc staging = {};
        createAlloc(size, D3D12_HEAP_TYPE_UPLOAD, &staging);

        // copy to staging
        char* pCpuPointer;
        checkResult(staging.pResource->Map(0, nullptr, reinterpret_cast<void**>(&pCpuPointer)));
        memcpy(pCpuPointer, pData, size);
        staging.pResource->Unmap(0, nullptr);

        // schedule a copy from staging to the alloc
        m_pCL->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(pAlloc->pResource, 
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST));

        m_pCL->CopyBufferRegion(pAlloc->pResource, pAlloc->offset, staging.pResource, 0, size);

        m_pCL->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(pAlloc->pResource,
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));


        // wait for it to finish
        flushAndWait();

        staging.pResource->Release();
    }
    
    void downloadData(void *pData, D3D12Alloc *pAlloc, size_t size)
    {
        // create a staging alloc
        D3D12Alloc staging = {};
        createAlloc(size, D3D12_HEAP_TYPE_READBACK, &staging);

        // schedule a copy from the alloc to staging
        m_pCL->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(pAlloc->pResource,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE));

        m_pCL->CopyBufferRegion(staging.pResource, 0, pAlloc->pResource, pAlloc->offset, size);

        m_pCL->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(pAlloc->pResource,
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

        // wait for it to finish
        flushAndWait();

        // copy from staging
        char* pCpuPointer;
        checkResult(staging.pResource->Map(0, nullptr, reinterpret_cast<void**>(&pCpuPointer)));
        memcpy(pData, pCpuPointer, size);
        staging.pResource->Unmap(0, nullptr);

        staging.pResource->Release();
    }

    void destroyAlloc(D3D12Alloc *pAlloc)
    {
        pAlloc->pResource->Release();
    }

    void dumpTensor(D3D12Alloc alloc, int size, bool fp16 = true, bool allnewline = false) {
        int bytes = size * (fp16 ? sizeof(uint16_t) : sizeof(float));

        void *data = malloc(bytes);
        downloadData(data, &alloc, bytes);

        printf("\n");

        float* fp32arr = (float*)data;
        uint16_t* arr = (uint16_t*)data;

        for (int i = 0; i < size; i++) {
            printf("%8.4f ", fp16 ? Fp16ToFp32(arr[i]) : fp32arr[i]);
            if (allnewline || ((i % 8) == 7)) printf("\n");
        }
        printf("\n");
        free(data);
    }

    void loadTensor(D3D12Alloc alloc, int size, bool fp16 = true)
    {
        int bytes = size * (fp16 ? sizeof(uint16_t) : sizeof(float));
        void *data = malloc(bytes);
        float* fp32arr = (float*)data;
        uint16_t* arr = (uint16_t*)data;

        for (int i = 0; i < size; i++) {
            float val;
            scanf_s("%f", &val);
            if (fp16)
                arr[i] = Fp32ToFp16(val);
            else
                fp32arr[i] = val;
        }
    }

    void destroy()
    {
        m_pFence->Release();
        m_pDescHeap->Release();
        m_queryResult.pResource->Release();
        m_pQueryHeap->Release();
        m_pCA->Release();
        m_pCQ->Release();
        m_pCL->Release();
        m_pDevice->Release();
    }
};

class ShaderWrapper {
private:
    // common for all shaders
    ID3D12RootSignature* m_pRootSign;

    ID3D12PipelineState* m_pInputTransformState;
    ID3D12PipelineState* m_pOutputTransformState;
    ID3D12PipelineState* m_pMatMulState;
	
public:
    void init(ID3D12Device* pDevice, bool fp16)
    {
        // 1. Create root signature - common for all shaders

        // 5 slots
        // slot 0 to 3 -> root UAV slots 0 to 3 (all in space 0), 
        // slot 4      -> root constants (16 constants - should be enough)
        // slot 5 to 8 -> desc heap UAVs (slots 5 to 8), uav slot no 4 unused.
        D3D12_DESCRIPTOR_RANGE descRange[4] = {};
        D3D12_ROOT_PARAMETER rootParameter[9];
        for (int i = 0; i < 4; i++) {
            rootParameter[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            rootParameter[i].Descriptor.RegisterSpace = 0;
            rootParameter[i].Descriptor.ShaderRegister = i;
            rootParameter[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        }

        rootParameter[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        rootParameter[4].Constants.RegisterSpace = 0;
        rootParameter[4].Constants.ShaderRegister = 0;
        rootParameter[4].Constants.Num32BitValues = 16;
        rootParameter[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        for (int i = 0; i < 4; i++) {
            descRange[i].BaseShaderRegister = i + 5;
            descRange[i].NumDescriptors = 1;
            descRange[i].OffsetInDescriptorsFromTableStart = 0;
            descRange[i].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            descRange[i].RegisterSpace = 0;

            rootParameter[i+5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParameter[i+5].DescriptorTable.NumDescriptorRanges = 1;
            rootParameter[i+5].DescriptorTable.pDescriptorRanges = &descRange[i];
            rootParameter[i+5].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        }


        D3D12_ROOT_SIGNATURE_DESC rootSigDesc = { 9, rootParameter, 0, NULL,
                                                 D3D12_ROOT_SIGNATURE_FLAG_NONE };

        ID3DBlob* pSerializedLayout = NULL;
        D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1,
            &pSerializedLayout, NULL);

        checkResult(pDevice->CreateRootSignature(
            1, pSerializedLayout->GetBufferPointer(),
            pSerializedLayout->GetBufferSize(), IID_PPV_ARGS(&m_pRootSign)));

        pSerializedLayout->Release();

        // Create PSO objects for each shader
        // PSO basically holds the compiled shader object (and other state which we
        // don't use)

        // 2. PSO for the InputTransform shader
        D3D12_COMPUTE_PIPELINE_STATE_DESC stateDesc = {};
        stateDesc.pRootSignature = m_pRootSign;

        if (fp16)
        {
            stateDesc.CS = { g_InputTransform_FP16,
                            sizeof(g_InputTransform_FP16) };
        }
        else
        {
            stateDesc.CS = { g_InputTransform_FP32,
                            sizeof(g_InputTransform_FP32) };
        }
        checkResult(pDevice->CreateComputePipelineState(
            &stateDesc, IID_PPV_ARGS(&m_pInputTransformState)));

        // 3. PSO for the OutputTransform shader
        if (fp16)
        {
            stateDesc.CS = { g_OutputTransform_FP16,
                            sizeof(g_OutputTransform_FP16) };
        }
        else
        {
            stateDesc.CS = { g_OutputTransform_FP32,
                            sizeof(g_OutputTransform_FP32) };
        }
        checkResult(pDevice->CreateComputePipelineState(
            &stateDesc, IID_PPV_ARGS(&m_pOutputTransformState)));

        // Ankan - test!
        // PSO for Matrix multiply shader			
        // fp16 = false;   // Ankan - hack: always use fp32 shader
        stateDesc.CS = { fp16 ? g_MatrixMul_Fp16 : g_MatrixMul_Fp32,
                        fp16 ? sizeof(g_MatrixMul_Fp16) : sizeof(g_MatrixMul_Fp32) };
        checkResult(pDevice->CreateComputePipelineState(
            &stateDesc, IID_PPV_ARGS(&m_pMatMulState)));
			
    }

    void destroy()
    {
        m_pMatMulState->Release();	
        m_pOutputTransformState->Release();
        m_pInputTransformState->Release();
        m_pRootSign->Release();
    }

    void InputTransform(int N, int C, ID3D12GraphicsCommandList *pCL, D3D12Alloc *pTransformedInput, D3D12Alloc *pInput)
    {
        int Consts[] = { N, C };
        pCL->SetComputeRootSignature(m_pRootSign);
        pCL->SetPipelineState(m_pInputTransformState);
        pCL->SetComputeRootUnorderedAccessView(0, pInput->gpuVA);
        pCL->SetComputeRootUnorderedAccessView(1, pTransformedInput->gpuVA);
        pCL->SetComputeRootDescriptorTable(5, pInput->descHandle);
        pCL->SetComputeRootDescriptorTable(6, pTransformedInput->descHandle);

        pCL->SetComputeRoot32BitConstants(4, 2, &Consts, 0);

        // TODO: remove hardcoding of 64
        int  blocks = divUp(N*C, 64);

        pCL->Dispatch(blocks, 1, 1);
    }

    void OutputTransform(int N, int K, ID3D12GraphicsCommandList *pCL, D3D12Alloc *pOutput, D3D12Alloc *TransformedOutput, D3D12Alloc *pBias, bool relu)
    {
        int Consts[] = { N, K, relu, !!pBias};
        pCL->SetComputeRootSignature(m_pRootSign);
        pCL->SetPipelineState(m_pOutputTransformState);
        pCL->SetComputeRootUnorderedAccessView(0, TransformedOutput->gpuVA);
        pCL->SetComputeRootUnorderedAccessView(1, pOutput->gpuVA);
        pCL->SetComputeRootUnorderedAccessView(2, pBias->gpuVA);
        pCL->SetComputeRootDescriptorTable(5, TransformedOutput->descHandle);
        pCL->SetComputeRootDescriptorTable(6, pOutput->descHandle);
        pCL->SetComputeRootDescriptorTable(7, pBias->descHandle);

        pCL->SetComputeRoot32BitConstants(4, 4, &Consts, 0);

        int  blocks = divUp(N*K, 64);
        pCL->Dispatch(blocks, 1, 1);
    }
	
    void MatrixMul(int M, int N, int K, int batch, ID3D12GraphicsCommandList *pCL, D3D12Alloc *pA, D3D12Alloc *pB, D3D12Alloc *pC)
    {
        int Consts[] = { M, N, K, batch };
        pCL->SetComputeRootSignature(m_pRootSign);
        pCL->SetPipelineState(m_pMatMulState);
        pCL->SetComputeRootDescriptorTable(5, pA->descHandle);
        pCL->SetComputeRootDescriptorTable(6, pB->descHandle);
        pCL->SetComputeRootDescriptorTable(7, pC->descHandle);
        pCL->SetComputeRoot32BitConstants(4, 4, &Consts, 0);
        int blocksX = divUp(N, ELEMENTS_PER_BLOCK_X);
        int blocksY = divUp(M, ELEMENTS_PER_BLOCK_Y);
        int blocksZ = batch;

        pCL->Dispatch(blocksX, blocksY, blocksZ);
    }	
};


D3d12Wrapper g_DXWrapper;
ShaderWrapper g_ShaderWrapper;

// get descriptor for row-major matrix (or batch of 'n' matrices)
static void getTensorDesc(TensorDesc* outDesc, int n, int rows, int cols, bool fp16 = true) 
{
    outDesc->DimensionCount = 4;
    outDesc->DataType = fp16 ? 1 : 0;

    outDesc->Size[0] = n;
    outDesc->Size[1] = 1;
    outDesc->Size[2] = rows;    // height
    outDesc->Size[3] = cols;    // width

    outDesc->Stride[3] = 1;
    outDesc->Stride[2] = cols;
    outDesc->Stride[1] = rows * cols;
    outDesc->Stride[0] = rows * cols;

    for (int i = 0; i < 4; i++) outDesc->StrideAlignment[i] = 1;
    outDesc->BaseAlignmentInBytes = 4096;
    outDesc->PhysicalSizeInElements = n * rows * cols;
}

void createGemmMetacommand(int M, int N, int K, int batch, bool useFp16, ID3D12MetaCommand **ppMetacommand)
{
    GemmCreateDesc createDesc = {};
    getTensorDesc(&createDesc.DescOut, batch, M, N, useFp16);
    getTensorDesc(&createDesc.DescA, batch, M, K, useFp16);
    getTensorDesc(&createDesc.DescB, batch, K, N, useFp16);
    createDesc.cMatrixNull = 1;
    createDesc.ActivationIsNull = 1;
    createDesc.Alpha = 1.0;
    createDesc.Beta = 0.0;
    createDesc.Precision = useFp16 ? 1 : 0; // 0 - fp32, 1 - fp16

    ID3D12MetaCommand *pMetacommand = nullptr;
    checkResult(g_DXWrapper.getDevice()->CreateMetaCommand(GemmGuid, 1, &createDesc, sizeof(createDesc), IID_PPV_ARGS(&pMetacommand)));
    *ppMetacommand = pMetacommand;
}

int main()
{
    const int gpuToUse = 0;
    g_DXWrapper.init(gpuToUse);
    bool fp16 = useFp16;

    g_ShaderWrapper.init(g_DXWrapper.getDevice(), fp16);

    constexpr int N = 256;
    constexpr int C = 256;
    constexpr int K = 256;

    // the paramaters below are hardcoded for our winograd transform kernels
    constexpr int H = 8;
    constexpr int W = 8;
    constexpr int F = 3;

    size_t elementSize = fp16 ? sizeof(uint16_t) : sizeof(float);
    size_t inputElements = N * C*H*W;
    size_t outputElements = N * K*H*W;
    size_t filterElements = K * C*F*F;
    size_t biasElements = K;

    size_t inputBytes = inputElements * elementSize;
    size_t outputBytes = outputElements * elementSize;
    size_t filterBytes = filterElements * elementSize;
    size_t biasBytes = biasElements * elementSize;
    size_t transformedFilterBytes = filterBytes * 4;
    size_t transformedInputBytes = inputBytes * (6 * 6) / (4 * 4);
    size_t transformedOutputBytes = outputBytes * (6 * 6) / (4 * 4);

    void *cinput = malloc(inputBytes);
    void *coutput = malloc(outputBytes);
    void *cpuRef = malloc(outputBytes);
    void *cfilter = malloc(filterBytes);
    void *cbias = malloc(biasBytes);
    void *ctransformedFilter = malloc(transformedFilterBytes);

    fillRandomArray(cinput, inputElements, fp16);
    fillRandomArray(cfilter, filterElements, fp16);
    fillRandomArray(cbias, biasElements, fp16);

    transformFilterTensor_Winograd4x4(K, C, ctransformedFilter, cfilter, fp16);

    D3D12Alloc input, output, transformedFilter, bias, transformedInput, transformedOutput;

    g_DXWrapper.createAlloc(inputBytes, D3D12_HEAP_TYPE_DEFAULT, &input);
    g_DXWrapper.createAlloc(outputBytes, D3D12_HEAP_TYPE_DEFAULT, &output);
    g_DXWrapper.createAlloc(transformedFilterBytes, D3D12_HEAP_TYPE_DEFAULT, &transformedFilter);
    g_DXWrapper.createAlloc(biasBytes, D3D12_HEAP_TYPE_DEFAULT, &bias);
    g_DXWrapper.createAlloc(transformedInputBytes, D3D12_HEAP_TYPE_DEFAULT, &transformedInput);
    g_DXWrapper.createAlloc(transformedOutputBytes, D3D12_HEAP_TYPE_DEFAULT, &transformedOutput);

    g_DXWrapper.uploadData(&input, cinput, inputBytes);
    g_DXWrapper.uploadData(&transformedFilter, ctransformedFilter, transformedFilterBytes);
    g_DXWrapper.uploadData(&bias, cbias, biasBytes);


    // Metacommand stuff
    ID3D12MetaCommand *pMetacommand = nullptr;
    GemmExecuteDesc execDesc = {};
    D3D12Alloc persistent = {}, temperory = {};
    if (useMetacommands)
    {
        createGemmMetacommand(N * 4, K, C, 36, fp16, &pMetacommand);
        size_t persistentSize = pMetacommand->GetRequiredParameterResourceSize(D3D12_META_COMMAND_PARAMETER_STAGE_EXECUTION, 4);
        size_t tempSize = pMetacommand->GetRequiredParameterResourceSize(D3D12_META_COMMAND_PARAMETER_STAGE_EXECUTION, 5);
        printf("\nPersistent size: %llu, temp size: %llu\n", persistentSize, tempSize);
        if (persistentSize)
            g_DXWrapper.createAlloc(persistentSize, D3D12_HEAP_TYPE_DEFAULT, &persistent);  // huge alloc - driver bug!
        if (tempSize)
            g_DXWrapper.createAlloc(tempSize, D3D12_HEAP_TYPE_DEFAULT, &persistent);


        GemmInitDesc initDesc = {};
        initDesc.PersistentResource = persistent.descHandle;
        g_DXWrapper.getCL()->InitializeMetaCommand(pMetacommand, &initDesc, sizeof(initDesc));
        g_DXWrapper.getCL()->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(persistent.pResource));

        execDesc.AResource = transformedInput.descHandle;
        execDesc.BResource = transformedFilter.descHandle;
        execDesc.OutputResource = transformedOutput.descHandle;
        execDesc.PersistentResource = persistent.descHandle;
        execDesc.TemporaryResource = temperory.descHandle;
    }

    int loops = 20;
    int iterPerLoop = 100;

    for (int i = 0; i < loops; i++)
    {
        g_DXWrapper.beginTimer();
        for (int j = 0; j < iterPerLoop; j++)
        {

            g_ShaderWrapper.InputTransform(N, C, g_DXWrapper.getCL(), &transformedInput, &input);

            //g_DXWrapper.dumpTensor(transformedInput, N*)

            g_DXWrapper.getCL()->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(transformedInput.pResource));
            if (useMetacommands)
                g_DXWrapper.getCL()->ExecuteMetaCommand(pMetacommand, &execDesc, sizeof(execDesc));
            else
                g_ShaderWrapper.MatrixMul(N * 4, K, C, 36, g_DXWrapper.getCL(), &transformedInput, &transformedFilter, &transformedOutput);
            g_DXWrapper.getCL()->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(transformedOutput.pResource));
            g_ShaderWrapper.OutputTransform(N, K, g_DXWrapper.getCL(), &output, &transformedOutput, &bias, true);

            g_DXWrapper.getCL()->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));   // wait for all UAVs
        }
        g_DXWrapper.endTimer();

        g_DXWrapper.flushAndWait();
        double time = g_DXWrapper.getTimeInSeconds();
        printf("\nTime taken: %g ms\n", time * 1000 / iterPerLoop);
    }

    g_DXWrapper.downloadData(coutput, &output, outputBytes);

    convRef(N, K, C, H, W, 3, 3, fp16, cpuRef, cinput, cfilter, cbias, true);
    compareResults(coutput, cpuRef, outputElements, fp16);
    

    if (persistent.pResource)
        g_DXWrapper.destroyAlloc(&persistent);

    if (temperory.pResource)
        g_DXWrapper.destroyAlloc(&temperory);

    g_DXWrapper.destroyAlloc(&input);
    g_DXWrapper.destroyAlloc(&output);
    g_DXWrapper.destroyAlloc(&transformedFilter);
    g_DXWrapper.destroyAlloc(&transformedInput);
    g_DXWrapper.destroyAlloc(&transformedOutput);
    g_ShaderWrapper.destroy();
    g_DXWrapper.destroy();

    free(cinput);
    free(cfilter);
    free(ctransformedFilter);
    free(coutput);
    free(cbias);

    getchar();
}