import { useAppState } from "../context/AppStateContext";
import SectionHeader from "../components/layout/SectionHeader";
import { ParchmentCard } from "../components/shared";

export default function SettingsPage() {
  const { apiKey, setApiKey, requireApiKey } = useAppState();

  return (
    <section className="max-w-xl mx-auto p-4 space-y-4">
      <SectionHeader title="Settings" />
      <ParchmentCard title="API Key">
        <p className="text-sm text-[var(--text-2)] mb-2">
          {requireApiKey
            ? "The server requires an API key. Enter it below to use Co-DM, voice, and AI features."
            : "Optional. If the server is configured to require an API key, enter it here."}
        </p>
        <label className="field-wrap block">
          <span>API Key</span>
          <input
            type="password"
            className="chat-input w-full"
            placeholder="Enter API key"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            autoComplete="off"
          />
        </label>
      </ParchmentCard>
      <p className="text-xs text-[var(--text-2)]">
        API key is stored in memory and used for all requests. Other preferences (e.g. default voice) can be added here.
      </p>
    </section>
  );
}
