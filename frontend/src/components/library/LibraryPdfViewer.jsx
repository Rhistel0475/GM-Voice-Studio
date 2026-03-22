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
 * Groups text layer spans into paragraph blocks; a new paragraph starts when gap > 1.5× line height.
 * Wraps each group in a .pdf-para-block div (preserving span positions) for hover/click.
 * @param {HTMLDivElement} textLayerDiv
 * @param {(text: string) => void} [onParagraphClick]
 */
function groupTextLayerIntoParagraphs(textLayerDiv, onParagraphClick) {
  const spans = Array.from(textLayerDiv.querySelectorAll("span")).filter((s) => s.textContent?.trim());
  if (spans.length === 0) return;

  const containerRect = textLayerDiv.getBoundingClientRect();
  /** @type {{ span: Element; top: number; bottom: number; left: number; right: number }[]} */
  const items = spans.map((span) => {
    const r = span.getBoundingClientRect();
    return {
      span,
      top: r.top - containerRect.top,
      bottom: r.bottom - containerRect.top,
      left: r.left - containerRect.left,
      right: r.right - containerRect.left,
    };
  });
  items.sort((a, b) => a.top - b.top || a.left - b.left);

  const firstHeight = items[0]?.bottom - items[0]?.top || 14;
  const gapThreshold = firstHeight * 1.2 * 1.5;

  /** @type {{ span: Element; top: number; bottom: number; left: number; right: number }[][]} */
  const groups = [];
  let current = [items[0]];
  let lastBottom = items[0].bottom;

  for (let i = 1; i < items.length; i++) {
    const it = items[i];
    if (it.top - lastBottom > gapThreshold) {
      groups.push(current);
      current = [];
    }
    current.push(it);
    lastBottom = it.bottom;
  }
  if (current.length) groups.push(current);

  groups.forEach((groupItems) => {
    const minL = Math.min(...groupItems.map((i) => i.left));
    const minT = Math.min(...groupItems.map((i) => i.top));
    const maxR = Math.max(...groupItems.map((i) => i.right));
    const maxB = Math.max(...groupItems.map((i) => i.bottom));

    const div = document.createElement("div");
    div.className = "pdf-para-block";
    div.style.cssText =
      "position:absolute;cursor:pointer;border-radius:4px;padding:2px 4px;transition:background 0.15s;";
    div.style.left = `${minL}px`;
    div.style.top = `${minT}px`;
    div.style.width = `${maxR - minL}px`;
    div.style.height = `${maxB - minT}px`;

    groupItems.forEach(({ span, left, top }) => {
      span.style.position = "absolute";
      span.style.left = `${left - minL}px`;
      span.style.top = `${top - minT}px`;
      span.style.margin = "0";
      span.style.transform = "none";
      div.appendChild(span);
    });

    if (onParagraphClick) {
      div.addEventListener("click", (e) => {
        e.stopPropagation();
        const text = Array.from(div.querySelectorAll("span"))
          .map((s) => s.textContent)
          .join(" ")
          .trim();
        if (text) onParagraphClick(text);
      });
    }
    textLayerDiv.appendChild(div);
  });
}

/**
 * Renders a PDF from a File (ArrayBuffer) with canvas + selectable text layer.
 * @param {{
 * file: File | null,
 * scale?: number,
 * containerWidth?: number,
 * onMeta?: (numPages: number) => void,
 * currentPage?: number,
 * onFitScaleChange?: (scale: number) => void,
 * onParagraphClick?: (text: string) => void
 * }} props
 */
const LibraryPdfViewer = forwardRef(function LibraryPdfViewer(
  { file, scale = 1, containerWidth = 0, currentPage = 1, onMeta, onFitScaleChange, onParagraphClick },
  ref
) {
  const hostRef = useRef(null);
  const canvasRef = useRef(null);
  const textLayerHostRef = useRef(null);
  const pdfRef = useRef(null);
  const pdfjsRef = useRef(null);
  const renderTaskRef = useRef(null);
  const renderPageRef = useRef(null);
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
      getCanvas() {
        return canvasRef.current;
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

        groupTextLayerIntoParagraphs(textLayerDiv, onParagraphClick);
      } catch (e) {
        if (e?.name !== "RenderingCancelledException") {
          throw e;
        }
      }
    },
    [scale, onParagraphClick]
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
        onMeta?.(pdf.numPages, pdf);
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
      </div>
    </div>
  );
});

export default LibraryPdfViewer;
