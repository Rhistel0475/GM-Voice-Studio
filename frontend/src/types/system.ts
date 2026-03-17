export interface CampaignSystemTerminology {
  skill_term: string;
  check_term: string;
  difficulty_term: string;
}

export interface CampaignSystemPreset {
  id: string;
  label: string;
  rules_flavor: string;
  skill_check_terminology: CampaignSystemTerminology;
  encounter_assumptions: string;
  thematic_guidance: string;
}
