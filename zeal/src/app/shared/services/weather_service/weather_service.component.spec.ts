import { ComponentFixture, TestBed } from '@angular/core/testing';

import { WeatherServiceComponent } from './weather-service.component';

describe('WeatherServiceComponent', () => {
  let component: WeatherServiceComponent;
  let fixture: ComponentFixture<WeatherServiceComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [WeatherServiceComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(WeatherServiceComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
