import { useAppState } from "../context/AppStateContext";
import VoiceStudioScreen from "../components/voices/VoiceStudioScreen";

export default function VoiceStudioPage({ campaignData: campaignDataProp, authFetch: authFetchProp }) {
  const appState = useAppState();
  const campaignData = campaignDataProp ?? appState.campaignData;
  const authFetch = authFetchProp ?? appState.authFetch;

  return (
    <VoiceStudioScreen
      campaignData={campaignData}
      authFetch={authFetch}
    />
  );
}
