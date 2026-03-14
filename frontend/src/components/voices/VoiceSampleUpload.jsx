import { useState, useRef, useEffect } from "react";
import { Upload, Mic, Play, Check, Trash2 } from "lucide-react";
import { FantasyButton } from "../shared";

function fmtTime(s) {
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

/**
 * Step 1: Upload an audio sample or record from microphone.
 * Recording produces a File passed through onFileChange, same as upload.
 */
export default function VoiceSampleUpload({
  file,
  onFileChange,
  name,
  onNameChange,
  onNext,
  disabled,
}) {
  // recState: idle | requesting | recording | recorded | error
  const [recState, setRecState] = useState("idle");
  const [recError, setRecError] = useState("");
  const [recBlob, setRecBlob] = useState(null);
  const [recUrl, setRecUrl] = useState(null);
  const [recSeconds, setRecSeconds] = useState(0);
  const [isPlayingRec, setIsPlayingRec] = useState(false);
  const mediaRecRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const recUrlRef = useRef(null);
  const audioRef = useRef(null);

  useEffect(() => {
    return () => {
      clearInterval(timerRef.current);
      if (recUrlRef.current) URL.revokeObjectURL(recUrlRef.current);
    };
  }, []);

  async function startRecording() {
    setRecState("requesting");
    setRecError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      mr.onstop = () => {
        clearInterval(timerRef.current);
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const url = URL.createObjectURL(blob);
        recUrlRef.current = url;
        setRecBlob(blob);
        setRecUrl(url);
        setRecState("recorded");
        stream.getTracks().forEach((t) => t.stop());
      };
      mr.start();
      mediaRecRef.current = mr;
      setRecSeconds(0);
      timerRef.current = setInterval(() => setRecSeconds((s) => s + 1), 1000);
      setRecState("recording");
    } catch (e) {
      setRecError(e?.message || "Microphone access denied.");
      setRecState("error");
    }
  }

  function stopRecording() {
    mediaRecRef.current?.stop();
  }

  function playRecording() {
    if (!audioRef.current) return;
    if (isPlayingRec) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setIsPlayingRec(false);
    } else {
      audioRef.current.play()
        .then(() => setIsPlayingRec(true))
        .catch(() => setIsPlayingRec(false));
    }
  }

  function useRecording() {
    if (!recBlob) return;
    onFileChange(new File([recBlob], "recording.webm", { type: "audio/webm" }));
  }

  function discardRecording() {
    if (recUrlRef.current) { URL.revokeObjectURL(recUrlRef.current); recUrlRef.current = null; }
    setRecBlob(null);
    setRecUrl(null);
    setRecState("idle");
    setRecSeconds(0);
    setIsPlayingRec(false);
    mediaRecRef.current = null;
    chunksRef.current = [];
  }

  const usingRecording = file?.name === "recording.webm";

  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm text-[var(--text-2)]">
        Upload a clear audio sample (WAV or MP3, 3–120 seconds). One speaker works best.
      </p>

      {/* File upload */}
      <label className="field-wrap">
        <span>Audio file</span>
        <div className="flex items-center gap-2">
          <input
            type="file"
            accept="audio/*"
            className="chat-input flex-1"
            onChange={(e) => {
              onFileChange(e.target.files?.[0] || null);
              if (recState === "recorded") discardRecording();
            }}
            disabled={disabled}
          />
          <Upload size={18} className="text-[var(--text-2)] shrink-0" />
        </div>
      </label>

      {/* Divider */}
      <div className="flex items-center gap-2">
        <div className="h-px flex-1 bg-[#c4a46b]/40" />
        <span className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider">or record</span>
        <div className="h-px flex-1 bg-[#c4a46b]/40" />
      </div>

      {/* Recording states */}
      <div className="flex flex-col gap-2">

        {recState === "idle" && (
          <button
            type="button"
            className="flex items-center gap-2 self-start text-sm font-heading text-[var(--ink-1)] border border-[#8c6435] rounded px-3 py-1.5 cursor-pointer hover:border-[#a17a42] hover:bg-[rgba(0,0,0,0.06)] transition-colors disabled:opacity-50"
            onClick={startRecording}
            disabled={disabled}
          >
            <Mic size={15} />
            Record Voice
          </button>
        )}

        {recState === "requesting" && (
          <p className="text-xs text-[var(--text-2)] italic">Requesting microphone access…</p>
        )}

        {recState === "recording" && (
          <div className="flex items-center gap-3">
            <span
              className="inline-block w-2.5 h-2.5 rounded-full animate-pulse"
              style={{ background: "#b91c1c" }}
              aria-label="Recording"
            />
            <span className="text-sm font-mono text-[var(--ink-1)] tabular-nums">{fmtTime(recSeconds)}</span>
            <button
              type="button"
              className="flex items-center gap-1.5 text-sm font-heading border rounded px-3 py-1 cursor-pointer transition-colors"
              style={{ color: "#b91c1c", borderColor: "#b91c1c" }}
              onClick={stopRecording}
            >
              Stop Recording
            </button>
          </div>
        )}

        {recState === "recorded" && (
          <div className="flex flex-col gap-2">
            <p className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider">
              Recording — {fmtTime(recSeconds)}
            </p>
            <audio
              ref={audioRef}
              src={recUrl}
              onEnded={() => setIsPlayingRec(false)}
              className="hidden"
            />
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                className="flex items-center gap-1.5 text-xs font-heading border border-[#8c6435] text-[var(--ink-1)] rounded px-2.5 py-1 cursor-pointer hover:border-[#a17a42] hover:bg-[rgba(0,0,0,0.06)] transition-colors"
                onClick={playRecording}
              >
                <Play size={11} className="shrink-0" />
                {isPlayingRec ? "Playing…" : "Play Recording"}
              </button>

              {usingRecording ? (
                <span className="text-xs font-heading" style={{ color: "#2d6a38" }}>✓ In use</span>
              ) : (
                <button
                  type="button"
                  className="flex items-center gap-1.5 text-xs font-heading border rounded px-2.5 py-1 cursor-pointer transition-colors"
                  style={{ color: "#2d6a38", borderColor: "#5f7d63" }}
                  onClick={useRecording}
                >
                  <Check size={11} className="shrink-0" />
                  Use Recording
                </button>
              )}

              <button
                type="button"
                className="flex items-center gap-1.5 text-xs font-heading text-[var(--text-2)] border border-[#8c6435] rounded px-2.5 py-1 cursor-pointer hover:text-[#b91c1c] hover:border-[#b91c1c] transition-colors"
                onClick={discardRecording}
              >
                <Trash2 size={11} className="shrink-0" />
                Discard Recording
              </button>
            </div>
          </div>
        )}

        {recState === "error" && (
          <div className="flex items-center gap-3">
            <p className="text-xs" style={{ color: "#b91c1c" }}>
              {recError || "Microphone error."}
            </p>
            <button
              type="button"
              className="text-xs font-heading text-[var(--text-2)] underline cursor-pointer"
              onClick={() => setRecState("idle")}
            >
              Try again
            </button>
          </div>
        )}

      </div>

      {/* Voice name */}
      <label className="field-wrap">
        <span>Voice name (optional)</span>
        <input
          type="text"
          placeholder="e.g. Tavern Keeper"
          className="chat-input w-full"
          value={name}
          onChange={(e) => onNameChange(e.target.value)}
          disabled={disabled}
        />
      </label>

      <FantasyButton
        variant="primary"
        className="w-full"
        onClick={onNext}
        disabled={!file || disabled}
      >
        Validate & start training
      </FantasyButton>
    </div>
  );
}
