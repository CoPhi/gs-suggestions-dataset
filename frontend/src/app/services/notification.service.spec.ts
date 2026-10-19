import { TestBed } from '@angular/core/testing';

import { NotificationService } from './notification.service';

describe('NotificationService', () => {
  let service: NotificationService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(NotificationService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should add a danger alert when showDanger is called', () => {
    service.showDanger('Modello eliminato');
    expect(service.alerts().length).toBe(1);
    expect(service.alerts()[0].type).toBe('danger');
    expect(service.alerts()[0].message).toBe('Modello eliminato');
  });

  it('should add a success alert when showSuccess is called', () => {
    service.showSuccess('Suggerimenti pronti');
    expect(service.alerts().length).toBe(1);
    expect(service.alerts()[0].type).toBe('success');
    expect(service.alerts()[0].message).toBe('Suggerimenti pronti');
  });

  it('should remove an alert by id', () => {
    service.showAlert('Test alert', 'info', 0);
    const alertId = service.alerts()[0].id;
    service.removeAlert(alertId);
    expect(service.alerts().length).toBe(0);
  });
});
