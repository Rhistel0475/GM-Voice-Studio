import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
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
 * scrollRootRef: React.RefObject<HTMLElement | null>,
 * onMeta?: (numPages: number) => void,
 * onVisiblePageChange?: (page: number) => void,
 * onFitScaleChange?: (scale: number) => void
 * }} props
 */
const LibraryPdfViewer = forwardRef(function LibraryPdfViewer(
  { file, scale = 1, containerWidth = 0, scrollRootRef, onMeta, onVisiblePageChange, onFitScaleChange },
  ref
) {
  const hostRef = useRef(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const obsRef = useRef(null);

  useImperativeHandle(
    ref,
    () => ({
      /** @param {number} p */
      scrollToPage(p) {
        const el = hostRef.current?.querySelector(`[data-page="${p}"]`);
        el?.scrollIntoView({ behavior: "smooth", block: "start" });
      },
    }),
    []
  );

  useEffect(() => {
    const host = hostRef.current;
    if (!file || !host) return;

    let cancelled = false;
    host.innerHTML = "";
    setLoading(true);
    setError("");
    onMeta?.(0);
    obsRef.current?.disconnect();

    (async () => {
      try {
        const pdfjsLib = await ensurePdfJs();
        const buf = await file.arrayBuffer();
        if (cancelled) return;
        const pdf = await pdfjsLib.getDocument({ data: buf }).promise;
        if (cancelled) return;
        onMeta?.(pdf.numPages);
        const firstPage = await pdf.getPage(1);
        const baseViewport = firstPage.getViewport({ scale: 1 });
        const safePaneWidth = Math.max(1, Number(containerWidth) || 0);
        const fitScale = safePaneWidth > 1 ? safePaneWidth / Math.max(1, baseViewport.width) : 1;
        onFitScaleChange?.(fitScale);

        for (let p = 1; p <= pdf.numPages; p++) {
          const page = await pdf.getPage(p);
          const viewport = page.getViewport({ scale });

          const wrap = document.createElement("div");
          wrap.dataset.page = String(p);
          wrap.className = "library-pdf-page-wrap";
          wrap.style.cssText = `position:relative;margin:0 auto 16px;width:${viewport.width}px;`;

          const canvas = document.createElement("canvas");
          const ctx = canvas.getContext("2d");
          if (!ctx) throw new Error("Canvas 2D not available");
          canvas.width = viewport.width;
          canvas.height = viewport.height;
          canvas.style.display = "block";

          const textLayerDiv = document.createElement("div");
          textLayerDiv.className = "textLayer";
          textLayerDiv.style.cssText = `position:absolute;left:0;top:0;height:${viewport.height}px;width:${viewport.width}px;pointer-events:auto;`;

          await page.render({ canvasContext: ctx, viewport }).promise;
          if (cancelled) return;

          wrap.appendChild(canvas);

          try {
            const textContent = await page.getTextContent();
            const textDivs = [];
            const renderTask = pdfjsLib.renderTextLayer({
              textContent,
              container: textLayerDiv,
              viewport,
              textDivs,
            });
            await renderTask.promise;
            if (cancelled) return;
            wrap.appendChild(textLayerDiv);
          } catch (tlErr) {
            console.warn("PDF text layer failed", tlErr);
          }

          host.appendChild(wrap);
        }

        if (cancelled) return;
        const root = scrollRootRef?.current;
        if (root && host.childElementCount > 0) {
          obsRef.current?.disconnect();
          const pages = host.querySelectorAll(".library-pdf-page-wrap[data-page]");
          const obs = new IntersectionObserver(
            (entries) => {
              const hits = entries
                .filter((e) => e.isIntersecting)
                .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
              if (hits[0]?.target?.dataset?.page) {
                const pn = parseInt(hits[0].target.dataset.page, 10);
                if (!Number.isNaN(pn)) onVisiblePageChange?.(pn);
              }
            },
            { root, threshold: [0.12, 0.25, 0.45, 0.65] }
          );
          pages.forEach((el) => obs.observe(el));
          obsRef.current = obs;
          onVisiblePageChange?.(1);
        }
      } catch (e) {
        if (!cancelled) setError(e?.message || "Failed to render PDF");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
      obsRef.current?.disconnect();
      obsRef.current = null;
    };
  }, [file, scale, containerWidth, scrollRootRef, onMeta, onVisiblePageChange, onFitScaleChange]);

  if (!file) return null;

  return (
    <div className="library-pdf-root w-full">
      {loading ? (
        <p className="text-xs text-[#9c7a3a] font-heading m-0 animate-pulse">Rendering PDF…</p>
      ) : null}
      {error ? <p className="text-xs text-amber-700/90 m-0">{error}</p> : null}
      <div ref={hostRef} className="library-pdf-pages mx-auto" />
    </div>
  );
});

export default LibraryPdfViewer;
