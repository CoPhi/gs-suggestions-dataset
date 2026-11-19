import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';

import { ApiService } from './api.service';

describe('ApiService', () => {
  let service: ApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(),
        provideHttpClientTesting()
      ]
    });
    service = TestBed.inject(ApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should append session_id query param to generateSuggestion and generateBatchSuggestions when provided', () => {
    service.generateSuggestion('m1', 'τειχεσι[....]', 1, 5, { session_id: 'sess-123' }).subscribe();
    const singleReq = httpMock.expectOne(
      req => req.url === 'http://127.0.0.1:8000/predictions' && req.params.get('session_id') === 'sess-123'
    );
    expect(singleReq.request.method).toBe('GET');
    singleReq.flush({ predictions: [] });

    service.generateBatchSuggestions(['m1', 'm2'], 'τειχεσι[....]', 1, 5, { session_id: 'sess-batch' }).subscribe();
    const batchReq = httpMock.expectOne(
      req => req.url === 'http://127.0.0.1:8000/predictions/batch' && req.params.get('session_id') === 'sess-batch'
    );
    expect(batchReq.request.method).toBe('GET');
    batchReq.flush({ results: [] });

    service.generateSuggestion('m1', 'τειχεσι[....]', 1, 5).subscribe();
    const plainReq = httpMock.expectOne(
      req => req.url === 'http://127.0.0.1:8000/predictions' && !req.params.has('session_id')
    );
    plainReq.flush({ predictions: [] });
  });
});
