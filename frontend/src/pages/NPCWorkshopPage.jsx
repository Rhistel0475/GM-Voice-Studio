import { useAppState } from "../context/AppStateContext";
import NPCWorkshopScreen from "../components/npcs/NPCWorkshopScreen";

export default function NPCWorkshopPage({ campaignData: campaignDataProp, authFetch: authFetchProp }) {
  const appState = useAppState();
  const campaignData = campaignDataProp ?? appState.campaignData;
  const authFetch = authFetchProp ?? appState.authFetch;

  return (
    <NPCWorkshopScreen
      campaignData={campaignData}
      authFetch={authFetch}
    />
  );
}
