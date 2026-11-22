import { TestBed } from '@angular/core/testing';
import { CHAT_HISTORY_TTL_MS, ChatHistoryService, ChatHistorySession } from './chat-history.service';

describe('ChatHistoryService', () => {
  let service: ChatHistoryService;

  const makeSession = (chatId: string, updatedAt: string, title = 'Τεστ *...*'): ChatHistorySession => ({
    chatId,
    title,
    createdAt: updatedAt,
    updatedAt,
    turns: [
      {
        turnId: `turn-${chatId}`,
        sessionId: `sess-${chatId}`,
        createdAt: updatedAt,
        rawText: title,
        normalizedText: title.replace(/\*/g, ''),
        numTokens: 1,
        numPredictions: 5,
        isRefinement: false,
        parentTurnId: null,
        appliedHumanFeedback: '',
        appliedRejectedCandidates: [],
        modelBoxes: [
          {
            modelId: 'm1',
            modelLabel: 'Qwen2.5-7B-Instruct',
            modelType: 'RAG',
            suggestions: [{ sentence: 'Τεστ λητα', token_str: 'λητα', score: 0.9 }],
            pending: false,
            error: null,
            submittedFeedback: {},
          },
        ],
        activeModelId: 'm1',
        userSelectedTab: false,
        isCollapsed: false,
        turnRejectedCandidates: [],
        turnHumanFeedback: '',
      },
    ],
  });

  beforeEach(async () => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ChatHistoryService);
    await service.clearAllChats();
  });

  afterEach(async () => {
    await service.clearAllChats();
  });

  it('should save, retrieve and sort chats by updatedAt descending', async () => {
    const older = makeSession('chat-old', '2026-11-18T10:00:00.000Z', 'Παλαιὰ *...*');
    const newer = makeSession('chat-new', '2026-11-20T12:00:00.000Z', 'Νέα *...*');

    await service.saveChat(older);
    await service.saveChat(newer);

    const all = await service.getAllChats();
    expect(all.length).toBe(2);
    expect(all[0].chatId).toBe('chat-new');
    expect(all[1].chatId).toBe('chat-old');

    const fetched = await service.getChat('chat-old');
    expect(fetched?.title).toBe('Παλαιὰ *...*');
  });

  it('should delete a single chat and clear all chats', async () => {
    await service.saveChat(makeSession('c1', '2026-11-20T10:00:00.000Z'));
    await service.saveChat(makeSession('c2', '2026-11-20T11:00:00.000Z'));

    await service.deleteChat('c1');
    let all = await service.getAllChats();
    expect(all.map(c => c.chatId)).toEqual(['c2']);

    await service.clearAllChats();
    all = await service.getAllChats();
    expect(all.length).toBe(0);
  });

  it('should purge only chats older than the retention TTL', async () => {
    const nowMs = Date.parse('2026-11-22T17:00:00.000Z');
    const expiredIso = new Date(nowMs - CHAT_HISTORY_TTL_MS - 60_000).toISOString();
    const recentIso = new Date(nowMs - (3 * 24 * 60 * 60 * 1000)).toISOString();

    await service.saveChat(makeSession('expired-chat', expiredIso));
    await service.saveChat(makeSession('recent-chat', recentIso));

    const purgedCount = await service.purgeExpiredChats(CHAT_HISTORY_TTL_MS, nowMs);
    expect(purgedCount).toBe(1);

    const remaining = await service.getAllChats();
    expect(remaining.length).toBe(1);
    expect(remaining[0].chatId).toBe('recent-chat');
  });
});
