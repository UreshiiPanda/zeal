import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe, NgIf } from '@angular/common';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';



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

  ],
  templateUrl: './weather.component.html',
  styleUrl: './weather.component.css'
})

export class WeatherComponent {

  showWeather: boolean = false;
  zipCode: string = '';
  weatherData: any = {
    city: 'Dallas, TX 75201',
    temp: '25°C',
    humidity: '90',
    conditions: 'Partly cloudy'
  };

  getWeather() {
    // Perform weather API call based on the zipCode
    // Update the weatherData based on the API response
    this.showWeather = true;
  }

  enterNewZipCode() {
    this.showWeather = false;
    this.zipCode = '';
  }
}
