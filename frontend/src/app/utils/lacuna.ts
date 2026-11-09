/**
 * Utility di pre-processing delle lacune per il frontend.
 *
 * Convenzione UI:
 * - `*...*` / `*---*` = lacuna target, quella che il filologo vuole integrare.
 * - `[...]` / `[---]` = lacuna contestuale, già presente nel testo e non vincolante
 *   per la generazione (non deve essere confusa con il target).
 *
 * Il backend continua a ricevere il formato canonico con una sola lacuna target
 * delimitata da parentesi quadre; le lacune contestuali vengono convertite in
 * puntini sciolti, che i modelli trattano già come contesto non vincolante (UNK).
 */

// Lacuna target digitata dall'utente: asterischi con punti/trattini interni.
// Sono ammessi caratteri greci ai bordi (es. `*κα...τος*`), che vengono poi
// spostati all'esterno da `extractBoundaryCharactersFromLacuna`.
export const TARGET_LACUNA_REGEX =
  /\*\s*([\p{Script=Greek}\u0300-\u036F\x27’‘ʼ᾽᾿´`]*?)\s*(\.{1,}|-{1,})\s*([\p{Script=Greek}\u0300-\u036F\x27’‘ʼ᾽᾿´`]*?)\s*\*/u;

// Lacuna contestuale in parentesi quadre: solo punti, trattini, spazi o ellissi.
export const CONTEXT_LACUNA_REGEX = /\[[.\-\s…×]+\]/gu;

// Forma target globale: consente di preservare eventuali caratteri di bordo greci
// (es. `*κα...τος*` -> `κα[...]τος`) qualora la utility venga usata senza la
// preventiva `extractBoundaryCharactersFromLacuna`.
const TARGET_LACUNA_GLOBAL_REGEX = new RegExp(TARGET_LACUNA_REGEX.source, 'gu');

/**
 * Converte le lacune contestuali `[...]` / `[---]` in puntini sciolti,
 * preservando il numero di caratteri interni alla lacuna.
 */
export function convertContextLacunaeToDots(text: string): string {
  return text.replace(CONTEXT_LACUNA_REGEX, (match) => '.'.repeat(match.length - 2));
}

/**
 * Converte la lacuna target `*...*` / `*---*` nel formato canonico `[...]` / `[---]`
 * atteso dalle API.
 */
export function convertTargetLacunaeToBrackets(text: string): string {
  return text.replace(
    TARGET_LACUNA_GLOBAL_REGEX,
    (_match, prefix: string, inner: string, suffix: string) => `${prefix}[${inner}]${suffix}`,
  );
}

/**
 * Normalizza un testo grezzo della UI per l'invio alle API.
 *
 * Se è presente almeno una lacuna target con asterischi, le parentesi quadre
 * presenti nel testo sono per definizione lacune contestuali: vanno neutralizzate
 * PRIMA di convertire gli asterischi in `[...]`, altrimenti il backend le
 * scambierebbe per il target.
 *
 * Se non ci sono asterischi, il testo è già stato normalizzato (es. payload HITL):
 * le eventuali quadre rappresentano già il target e vengono lasciate intatte.
 */
export function normalizeLacunaeForApi(text: string): string {
  if (!text) return '';
  if (TARGET_LACUNA_REGEX.test(text)) {
    return convertTargetLacunaeToBrackets(convertContextLacunaeToDots(text));
  }
  return text;
}

/**
 * Rimuove le lacune contestuali valide, lasciando il resto del testo per la
 * validazione dei caratteri ammessi.
 */
export function stripContextLacunae(text: string): string {
  return text.replace(CONTEXT_LACUNA_REGEX, '');
}
