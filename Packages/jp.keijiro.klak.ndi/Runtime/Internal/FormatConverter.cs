using UnityEngine;
using UnityEngine.Rendering;
using IDisposable = System.IDisposable;
using IntPtr = System.IntPtr;

namespace Klak.Ndi {

//
// A format converter class wrapping GPU tasks and resources
//
// We use GPU (compute) to convert textures between the renderer-friendly raw
// formats and the NDI-friendly (chroma-subsampled) formats. This class wraps
// the cumbersome things and provides a simple API.
//
sealed class FormatConverter : IDisposable
{
    #region Common members

    NdiResources _resources;

    public FormatConverter(NdiResources resources) => _resources = resources;

    public void Dispose() => ReleaseBuffers();

    void ReleaseBuffers()
    {
        _encoderOutput?.Dispose();
        _encoderOutput = null;

        _decoderInput?.Dispose();
        _decoderInput = null;

        Util.Destroy(_decoderOutput);
        _decoderOutput = null;
    }

    // The only dimension requirement left is the horizontal pixel pairing
    // imposed by the 4:2:2 chroma subsampling of the NDI wire formats. Senders
    // trim odd widths via Util.AlignWidth, so this is a safety net.
    void CheckDimensions(int width, int height)
    {
        if ((width & 0x1) != 0)
            WarnWrongSize($"Width ({width}) must be an even number.");
    }

    void WarnWrongSize(string text)
      => Debug.LogWarning("[KlakNDI] Unsupported frame size: " + text);

    #endregion

    #region Encoder implementation

    ComputeBuffer _encoderOutput;

    // Kernel index of the alpha plane pass in the encoder compute shader
    const int AlphaPass = 2;

    // Thread grid width of the alpha pass, in threads. The alpha plane is
    // addressed as a flat array, so its grid is folded in 2D to stay within
    // the per-dimension dispatch limit on large frames.
    const int AlphaGridWidth = 512;

    static int EncoderPass => Util.InGammaMode ? 0 : 1;

    static (int x, int y) AlphaDispatchSize(int width, int height)
    {
        var count = (width * height + 3) / 4;
        var rows = (count + AlphaGridWidth - 1) / AlphaGridWidth;
        return (AlphaGridWidth / 8, (rows + 7) / 8);
    }

    // Immediate mode version
    public ComputeBuffer Encode
      (Texture source, int width, int height, bool enableAlpha, bool vflip)
    {
        var dataCount = Util.FrameDataSize(width, height, enableAlpha) / 4;

        // Reallocate the output buffer when the output size was changed.
        if (_encoderOutput != null && _encoderOutput.count != dataCount)
            ReleaseBuffers();

        // Output buffer allocation
        if (_encoderOutput == null)
        {
            CheckDimensions(width, height);
            _encoderOutput = new ComputeBuffer(dataCount, 4);
        }

        // Compute thread dispatching
        //
        // A thread covers a horizontal pixel pair and a group covers 16x8
        // pixels, so the dispatch size is rounded up to keep the right and
        // bottom edges of unaligned frames covered.
        var compute = _resources.encoderCompute;
        var pass = EncoderPass;
        compute.SetFloat("VFlip", vflip ? 1 : 0);
        SetGeometry(compute, width, height);
        compute.SetTexture(pass, "Source", source);
        compute.SetBuffer(pass, "Destination", _encoderOutput);
        compute.Dispatch(pass, (width + 15) / 16, (height + 7) / 8, 1);

        // Alpha plane pass
        if (enableAlpha)
        {
            var (gx, gy) = AlphaDispatchSize(width, height);
            compute.SetTexture(AlphaPass, "Source", source);
            compute.SetBuffer(AlphaPass, "Destination", _encoderOutput);
            compute.Dispatch(AlphaPass, gx, gy, 1);
        }

        return _encoderOutput;
    }

    // Command buffer version
    public ComputeBuffer Encode
      (CommandBuffer cb, RenderTargetIdentifier source,
       int width, int height, bool enableAlpha, bool vflip)
    {
        var dataCount = Util.FrameDataSize(width, height, enableAlpha) / 4;

        // Reallocate the output buffer when the output size was changed.
        if (_encoderOutput != null && _encoderOutput.count != dataCount)
            ReleaseBuffers();

        // Output buffer allocation
        if (_encoderOutput == null)
        {
            CheckDimensions(width, height);
            _encoderOutput = new ComputeBuffer(dataCount, 4);
        }

        // Compute thread dispatching
        var compute = _resources.encoderCompute;
        var pass = EncoderPass;
        cb.SetComputeFloatParam(compute, "VFlip", vflip ? 1 : 0);
        SetGeometry(cb, compute, width, height);
        cb.SetComputeTextureParam(compute, pass, "Source", source);
        cb.SetComputeBufferParam(compute, pass, "Destination", _encoderOutput);
        cb.DispatchCompute(compute, pass, (width + 15) / 16, (height + 7) / 8, 1);

        // Alpha plane pass
        if (enableAlpha)
        {
            var (gx, gy) = AlphaDispatchSize(width, height);
            cb.SetComputeTextureParam(compute, AlphaPass, "Source", source);
            cb.SetComputeBufferParam(compute, AlphaPass, "Destination", _encoderOutput);
            cb.DispatchCompute(compute, AlphaPass, gx, gy, 1);
        }

        return _encoderOutput;
    }

    // Encoder frame geometry
    //
    // The output is tightly packed: the UYVY plane uses a width*2 line stride
    // and the alpha plane directly follows it.
    void SetGeometry(ComputeShader compute, int width, int height)
    {
        compute.SetInt("FrameWidth", width);
        compute.SetInt("FrameHeight", height);
        compute.SetInt("LineStride", width / 2);
        compute.SetInt("AlphaOffset", width * height / 2);
        compute.SetInt("AlphaCount", (width * height + 3) / 4);
        compute.SetInt("AlphaGridWidth", AlphaGridWidth);
    }

    void SetGeometry
      (CommandBuffer cb, ComputeShader compute, int width, int height)
    {
        cb.SetComputeIntParam(compute, "FrameWidth", width);
        cb.SetComputeIntParam(compute, "FrameHeight", height);
        cb.SetComputeIntParam(compute, "LineStride", width / 2);
        cb.SetComputeIntParam(compute, "AlphaOffset", width * height / 2);
        cb.SetComputeIntParam(compute, "AlphaCount", (width * height + 3) / 4);
        cb.SetComputeIntParam(compute, "AlphaGridWidth", AlphaGridWidth);
    }

    #endregion

    #region Decoder implementation

    ComputeBuffer _decoderInput;
    RenderTexture _decoderOutput;

    public RenderTexture LastDecoderOutput => _decoderOutput;

    public RenderTexture Decode
      (int width, int height, bool enableAlpha, int lineStride, IntPtr data)
    {
        // Line stride fallback: senders may leave it unspecified when the
        // frame is tightly packed.
        if (lineStride <= 0) lineStride = width * 2;

        // The frame is uploaded as an array of uints, so each line of the UYVY
        // plane has to start on a 4-byte boundary. Any even width satisfies
        // this, and NDI doesn't emit odd widths for subsampled formats.
        if ((lineStride & 0x3) != 0)
        {
            WarnWrongSize($"Line stride ({lineStride}) must be a multiple of 4.");
            return _decoderOutput;
        }

        // Alpha plane: it directly follows the UYVY plane and carries one byte
        // per pixel, hence half the line stride.
        var alphaOffset = enableAlpha ? lineStride * height : 0;
        var alphaStride = enableAlpha ? lineStride / 2 : 0;

        // Frame size in uints. The rounding only kicks in on an alpha frame
        // that is both odd-height and half-aligned, which no known sender
        // produces.
        var dataCount = (lineStride * height + alphaStride * height + 3) / 4;

        // Reallocate the input buffer when the input size was changed.
        if (_decoderInput != null && _decoderInput.count != dataCount)
            ReleaseBuffers();

        // Reallocate the output buffer when the output size was changed.
        if (_decoderOutput != null &&
            (_decoderOutput.width != width ||
             _decoderOutput.height != height))
            ReleaseBuffers();

        // Input buffer allocation
        if (_decoderInput == null)
            _decoderInput = new ComputeBuffer(dataCount, 4);

        // Output buffer allocation
        if (_decoderOutput == null)
        {
            CheckDimensions(width, height);
        #if KLAK_NDI_ISSUE200_WORKAROUND
            _decoderOutput = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBHalf);
        #else
            _decoderOutput = new RenderTexture(width, height, 0);
        #endif
            _decoderOutput.enableRandomWrite = true;
            _decoderOutput.Create();
        }

        // Input buffer update
        _decoderInput.SetData(data, dataCount, 4);

        // Kenel select
        var pass = (enableAlpha ? 2 : 0);

        // As far as we know, only Metal supports sRGB write to UAV, so we
        // use the linear-color kernels only on Metal
        if (!Util.InGammaMode && Util.UsingMetal) pass++;

        // Decoder compute dispatching
        //
        // A thread covers a horizontal pixel pair and a group covers 16x8
        // pixels, so the dispatch size is rounded up to keep the right and
        // bottom edges of unaligned frames covered.
        var compute = _resources.decoderCompute;
        compute.SetInt("FrameWidth", width);
        compute.SetInt("FrameHeight", height);
        compute.SetInt("LineStride", lineStride / 4);
        compute.SetInt("AlphaOffset", alphaOffset);
        compute.SetInt("AlphaStride", alphaStride);
        compute.SetBuffer(pass, "Source", _decoderInput);
        compute.SetTexture(pass, "Destination", _decoderOutput);
        compute.Dispatch(pass, (width + 15) / 16, (height + 7) / 8, 1);

        return _decoderOutput;
    }

    #endregion
}

} // namespace Klak.Ndi
