
import { RouterModule, RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { Component } from '@angular/core';
import { OnInit } from '@angular/core';
import { initFlowbite } from 'flowbite';
import { HeaderComponent } from './shared/components/header/header.component';
import { FooterComponent } from './shared/components/footer/footer.component';
import { TipsComponent } from './shared/components/tips/tips.component';
import { WeatherComponent } from './shared/components/weather/weather.component';
import { RpsComponent } from './shared/components/rps/rps.component';
import { GroceryComponent } from './shared/components/grocery/grocery.component';
import { HomeComponent } from './shared/components/home/home.component';


@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    RouterModule,
    RouterOutlet,
    RouterLink,
    RouterLinkActive,
    HomeComponent,
    HeaderComponent,
    FooterComponent,
    TipsComponent,
    GroceryComponent,
    WeatherComponent,
    RpsComponent
  ],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'zeal';

  ngOnInit(): void {
    initFlowbite();
  }
}

