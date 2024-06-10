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

  getWeather() {
    // Perform weather API call based on the zipCode
    // Update the weatherData based on the API response

    this.WeatherServiceComponent.getWeatherByZipCode(this.zipCode).subscribe(
      data => {
        this.weatherData = data;
      },
      error => {
        console.error('Error fetching weather data:', error);
      }
    );
    this.showWeather = true;
  }

  enterNewZipCode() {
    this.showWeather = false;
    this.zipCode = '';
  }
}

