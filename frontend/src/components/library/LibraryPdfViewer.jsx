import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
const PDFJS_URL = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
const WORKER_SRC = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

/** @type {Promise<typeof window.pdfjsLib> | null} */
let pdfjsLoadPromise = null;

function ensurePdfJs() {
  if (typeof window !== "undefined" && window.pdfjsLib) {
    window.pdfjsLib.GlobalWorkerOptions.workerSrc = WORKER_SRC;
    return Promise.resolve(window.pdfjsLib);
  }
  if (pdfjsLoadPromise) return pdfjsLoadPromise;
  pdfjsLoadPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = PDFJS_URL;
    script.async = true;
    script.onload = () => {
      const pdfjsLib = window.pdfjsLib;
      if (!pdfjsLib) {
        reject(new Error("pdfjsLib not available after load"));
        return;
      }
      pdfjsLib.GlobalWorkerOptions.workerSrc = WORKER_SRC;
      resolve(pdfjsLib);
    };
    script.onerror = () => reject(new Error("Failed to load PDF.js"));
    document.head.appendChild(script);
  });
  return pdfjsLoadPromise;
}

/**
 * Renders a PDF from a File (ArrayBuffer) with canvas + selectable text layer.
 * @param {{
 * file: File | null,
 * scale?: number,
 * containerWidth?: number,
 * onMeta?: (numPages: number) => void,
 * currentPage?: number,
 * onFitScaleChange?: (scale: number) => void
 * }} props
 */
const LibraryPdfViewer = forwardRef(function LibraryPdfViewer(
  { file, scale = 1, containerWidth = 0, currentPage = 1, onMeta, onFitScaleChange },
  ref
) {
  const hostRef = useRef(null);
  const canvasRef = useRef(null);
  const textLayerHostRef = useRef(null);
  const pdfRef = useRef(null);
  const pdfjsRef = useRef(null);
  const renderTaskRef = useRef(null);
  const renderPageRef = useRef(null);
  const selectionMirrorRef = useRef(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useImperativeHandle(
    ref,
    () => ({
      cancelRender() {
        if (renderTaskRef.current) {
          renderTaskRef.current.cancel();
          renderTaskRef.current = null;
        }
      },
      /** @param {string} text */
      mirrorSelectionText(text) {
        if (selectionMirrorRef.current) {
          selectionMirrorRef.current.value = text || "";
        }
      },
    }),
    []
  );

  const renderPage = useCallback(
    async (pageNumber) => {
      const canvas = canvasRef.current;
      const textLayerHost = textLayerHostRef.current;
      const pdfDoc = pdfRef.current;
      const pdfjsLib = pdfjsRef.current;
      if (!canvas || !textLayerHost || !pdfDoc || !pdfjsLib) return;

      if (renderTaskRef.current) {
        renderTaskRef.current.cancel();
        renderTaskRef.current = null;
      }

      const total = pdfDoc.numPages || 1;
      const pageNum = Math.max(1, Math.min(total, Number(pageNumber) || 1));
      textLayerHost.innerHTML = "";

      try {
        const page = await pdfDoc.getPage(pageNum);
        const viewport = page.getViewport({ scale });
        const ctx = canvas.getContext("2d");
        if (!ctx) throw new Error("Canvas 2D not available");

        canvas.width = viewport.width;
        canvas.height = viewport.height;
        canvas.style.width = `${viewport.width}px`;
        canvas.style.height = `${viewport.height}px`;
        if (hostRef.current) {
          hostRef.current.style.width = `${viewport.width}px`;
        }

        const renderTask = page.render({ canvasContext: ctx, viewport });
        renderTaskRef.current = renderTask;
        await renderTask.promise;

        const textLayerDiv = document.createElement("div");
        textLayerDiv.className = "textLayer";
        textLayerDiv.style.cssText = `position:absolute;left:0;top:0;height:${viewport.height}px;width:${viewport.width}px;pointer-events:auto;`;
        textLayerHost.appendChild(textLayerDiv);

        const textContent = await page.getTextContent();
        const textDivs = [];
        const textTask = pdfjsLib.renderTextLayer({
          textContent,
          container: textLayerDiv,
          viewport,
          textDivs,
        });
        await textTask.promise;
      } catch (e) {
        if (e?.name !== "RenderingCancelledException") {
          throw e;
        }
      }
    },
    [scale]
  );

  useEffect(() => {
    renderPageRef.current = renderPage;
  }, [renderPage]);

  useEffect(() => {
    const host = hostRef.current;
    const canvas = canvasRef.current;
    const textLayerHost = textLayerHostRef.current;
    if (!file || !host || !canvas || !textLayerHost) return;

    let cancelled = false;
    pdfRef.current = null;
    setLoading(true);
    setError("");
    onMeta?.(0);

    (async () => {
      try {
        const pdfjsLib = await ensurePdfJs();
        pdfjsRef.current = pdfjsLib;
        const buf = await file.arrayBuffer();
        if (cancelled) return;
        const pdf = await pdfjsLib.getDocument({ data: buf }).promise;
        pdfRef.current = pdf;
        if (cancelled) return;
        onMeta?.(pdf.numPages);
        const firstPage = await pdf.getPage(1);
        const baseViewport = firstPage.getViewport({ scale: 1 });
        const safePaneWidth = Math.max(1, Number(containerWidth) || 0);
        const fitScale = safePaneWidth > 1 ? safePaneWidth / Math.max(1, baseViewport.width) : 1;
        onFitScaleChange?.(fitScale);
        await renderPageRef.current?.(1);
      } catch (e) {
        if (!cancelled && e?.name !== "RenderingCancelledException") {
          setError(e?.message || "Failed to render PDF");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
      if (renderTaskRef.current) {
        renderTaskRef.current.cancel();
        renderTaskRef.current = null;
      }
      pdfRef.current?.destroy();
      pdfRef.current = null;
      textLayerHost.innerHTML = "";
    };
  }, [file]);

  useEffect(() => {
    if (!file || !pdfRef.current) return;

    let cancelled = false;
    const run = async () => {
      try {
        setLoading(true);
        setError("");
        await renderPage(currentPage);
      } catch (e) {
        if (!cancelled && e?.name !== "RenderingCancelledException") {
          setError(e?.message || "Failed to render PDF page");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    void run();
    return () => {
      cancelled = true;
      if (renderTaskRef.current) {
        renderTaskRef.current.cancel();
        renderTaskRef.current = null;
      }
    };
  }, [currentPage, scale]);

  if (!file) return null;

  return (
    <div className="library-pdf-root w-full">
      {loading ? (
        <p className="text-xs text-[#e7c27a] font-heading m-0 animate-pulse">Loading page...</p>
      ) : null}
      {error ? <p className="text-xs text-amber-700/90 m-0">{error}</p> : null}
      <div ref={hostRef} className="library-pdf-page-wrap mx-auto relative">
        <canvas ref={canvasRef} className="block" />
        <div ref={textLayerHostRef} />
        <textarea
          ref={selectionMirrorRef}
          readOnly
          tabIndex={-1}
          aria-hidden
          className="absolute left-0 top-0 h-full w-full opacity-0 pointer-events-none resize-none border-0 p-0 m-0"
        />
      </div>
    </div>
  );
});

export default LibraryPdfViewer;
