import { Component } from '@angular/core';
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';


@Injectable({
  providedIn: 'root'
})


@Component({
  selector: 'app-weather-service',
  standalone: true,
  imports: [],
  templateUrl: './weather_service.component.html',
  styleUrl: './weather_service.component.css'
})
export class WeatherServiceComponent {
  private apiUrl = 'http://localhost:5006/api/weather';

  constructor(private http: HttpClient) { }

  getWeatherByZipCode(zipCode: string): Observable<any> {
    const url = `${this.apiUrl}?zipCode=${zipCode}`;
    console.log("Weather URL: ", url);
    console.log("Data from URL: ", this.http.get(url));
    return this.http.get(url);
  }
}



