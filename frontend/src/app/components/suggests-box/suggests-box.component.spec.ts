import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';

import { SuggestsBoxComponent } from './suggests-box.component';

describe('SuggestsBoxComponent', () => {
  let component: SuggestsBoxComponent;
  let fixture: ComponentFixture<SuggestsBoxComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SuggestsBoxComponent],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting()
      ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SuggestsBoxComponent);
    fixture.componentRef.setInput('suggestions', []);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should emit onClose when close() is called', () => {
    spyOn(component.onClose, 'emit');
    component.close();
    expect(component.onClose.emit).toHaveBeenCalled();
  });

  it('should emit onClose when close button is clicked in template', () => {
    spyOn(component.onClose, 'emit');
    const closeBtn = fixture.nativeElement.querySelector('.close-btn') as HTMLButtonElement;
    expect(closeBtn).toBeTruthy();
    closeBtn.click();
    expect(component.onClose.emit).toHaveBeenCalled();
  });
});
