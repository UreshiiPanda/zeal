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
  city: string = '';
  desc: string = '';
  humidity: GLfloat = 0;
  temp: number = 0;
  weatherData: any = {
    "city": '',
    "conditions": '',
    "temp": 0,
    "humidity": 0
  };

  constructor(private weatherService: WeatherServiceComponent) {}

  getWeather() {
    this.weatherService.getWeatherByZipCode(this.zipCode).subscribe(
      (data: any) => {
        this.weatherData = data;
        // console.log(this.weatherData);
        console.log(this.weatherData.name);
        console.log(this.weatherData.main.temp);
        console.log(this.weatherData.weather[0].description);
        console.log(this.weatherData.main.humidity);
        this.city = data.name;
        this.desc = data.weather[0].description;
        this.temp = data.main.temp;
        this.humidity = data.main.humidity;
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
    this.city = '';
    this.humidity = 0;
    this.temp = 0;
  }
}

