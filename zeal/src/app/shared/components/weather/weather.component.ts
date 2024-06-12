import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe, NgIf } from '@angular/common';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { WeatherServiceComponent } from '../../services/weather_service/weather_service.component';



@Component({
  selector: 'weather',
  standalone: true,
  imports: [
    NgIf,
    DecimalPipe,
    FormsModule,
    RouterOutlet,
    RouterLink,
    RouterLinkActive,
    WeatherServiceComponent,
  ],
  templateUrl: './weather.component.html',
  styleUrl: './weather.component.css'
})

export class WeatherComponent {

  showWeather: boolean = false;
  zipCode: string = '';
  weatherData: any = {};

  constructor(private weatherService: WeatherServiceComponent) {}

  getWeather() {
    this.weatherService.getWeatherByZipCode(this.zipCode).subscribe(
      (data: any) => {
        this.weatherData = data;
        this.showWeather = true;
      },
      (error: any) => {
        console.error('Error fetching weather data:', error);
      }
    );
  }
  enterNewZipCode() {
    this.showWeather = false;
    this.zipCode = '';
  }
}

