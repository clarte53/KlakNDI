using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Rendering;
using IntPtr = System.IntPtr;

namespace Klak.Ndi {

// Small utility functions
static class Util
{
    public static int FrameDataSize(int width, int height, bool alpha)
    {
        // UYVY plane (two bytes per pixel), followed by the alpha plane (one
        // byte per pixel) rounded up to the uint granularity the encoder
        // writes with. The rounding is a no-op unless the pixel count is not a
        // multiple of 4.
        var size = width * height * 2;
        if (alpha) size += ((width * height + 3) / 4) * 4;
        return size;
    }

    // 4:2:2 chroma subsampling transmits pixels in horizontal pairs, so an odd
    // width can't be represented. Sent frames are trimmed to the pair below.
    public static int AlignWidth(int width) => width & ~1;

    public static bool HasAlpha(Interop.FourCC fourCC)
      => fourCC == Interop.FourCC.UYVA;

    public static bool InGammaMode
      => QualitySettings.activeColorSpace == ColorSpace.Gamma;

    public static bool UsingMetal
      => SystemInfo.graphicsDeviceType == GraphicsDeviceType.Metal;

    public static void Destroy(Object obj)
    {
        if (obj == null) return;

        if (Application.isPlaying)
            Object.Destroy(obj);
        else
            Object.DestroyImmediate(obj);
    }
}

// Extension method to add IntPtr support to ComputeBuffer.SetData
static class ComputeBufferExtension
{
    public unsafe static void SetData
      (this ComputeBuffer buffer, IntPtr pointer, int count, int stride)
    {
        // NativeArray view for the unmanaged memory block
        var view =
          NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<byte>
            ((void*)pointer, count * stride, Allocator.None);

        #if ENABLE_UNITY_COLLECTIONS_CHECKS
        var safety = AtomicSafetyHandle.Create();
        NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref view, safety);
        #endif

        buffer.SetData(view);

        #if ENABLE_UNITY_COLLECTIONS_CHECKS
        AtomicSafetyHandle.Release(safety);
        #endif
    }
}

} // namespace Klak.Ndi
