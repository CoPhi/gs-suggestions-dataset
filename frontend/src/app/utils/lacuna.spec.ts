import {
  convertContextLacunaeToDots,
  convertTargetLacunaeToBrackets,
  formatNormalizedLacunaForPrompt,
  normalizeLacunaeForApi,
  splitLacunaHighlightSegments,
  stripContextLacunae,
} from './lacuna';

describe('lacuna utils', () => {
  it('should convert contextual square-bracket lacunae to loose dots', () => {
    expect(convertContextLacunaeToDots('τὰ [...] αὐτὰ [....] ὑπὲρ')).toBe('τὰ ... αὐτὰ .... ὑπὲρ');
  });

  it('should not treat hyphen square-bracket lacunae as contextual lacunae', () => {
    expect(convertContextLacunaeToDots('τὰ [---] αὐτὰ')).toBe('τὰ [---] αὐτὰ');
  });

  it('should convert the asterisk target lacuna to brackets', () => {
    expect(convertTargetLacunaeToBrackets('τὰ *...* αὐτὰ *....* ὑπὲρ')).toBe('τὰ [...] αὐτὰ [....] ὑπὲρ');
  });

  it('should not convert hyphen target lacunae (*---*) to brackets', () => {
    expect(convertTargetLacunaeToBrackets('τὰ *---* αὐτὰ')).toBe('τὰ *---* αὐτὰ');
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

  it('should convert normalized bracket lacuna back to asterisk syntax for prompt reuse', () => {
    expect(formatNormalizedLacunaForPrompt('τὰ ... αὐτὰ [...] ὑπὲρ')).toBe('τὰ ... αὐτὰ *...* ὑπὲρ');
  });

  it('should split text into plain and highlighted lacuna segments', () => {
    expect(splitLacunaHighlightSegments('τὰ [...] ὑπὲρ')).toEqual([
      { text: 'τὰ ', isLacuna: false },
      { text: '[...]', isLacuna: true },
      { text: ' ὑπὲρ', isLacuna: false },
    ]);
  });
});
