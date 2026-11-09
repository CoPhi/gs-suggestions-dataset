import {
  convertContextLacunaeToDots,
  convertTargetLacunaeToBrackets,
  normalizeLacunaeForApi,
  stripContextLacunae,
} from './lacuna';

describe('lacuna utils', () => {
  it('should convert contextual square-bracket lacunae to loose dots', () => {
    expect(convertContextLacunaeToDots('τὰ [...] αὐτὰ [---] ὑπὲρ')).toBe('τὰ ... αὐτὰ ... ὑπὲρ');
  });

  it('should convert the asterisk target lacuna to brackets', () => {
    expect(convertTargetLacunaeToBrackets('τὰ *...* αὐτὰ *---* ὑπὲρ')).toBe('τὰ [...] αὐτὰ [---] ὑπὲρ');
  });

  it('should neutralize context brackets before converting the target', () => {
    expect(normalizeLacunaeForApi('τὰ [...] αὐτὰ *...* ὑπὲρ')).toBe('τὰ ... αὐτὰ [...] ὑπὲρ');
  });

  it('should leave already normalized payloads unchanged', () => {
    expect(normalizeLacunaeForApi('τὰ ... αὐτὰ [...] ὑπὲρ')).toBe('τὰ ... αὐτὰ [...] ὑπὲρ');
  });

  it('should strip only valid contextual lacunae', () => {
    expect(stripContextLacunae('τὰ [...] αὐτὰ [κατ\'] *...* ὑπὲρ')).toBe('τὰ  αὐτὰ [κατ\'] *...* ὑπὲρ');
  });
});
