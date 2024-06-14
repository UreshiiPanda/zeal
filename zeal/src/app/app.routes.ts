import { RouterModule, Routes } from '@angular/router';
import { HeaderComponent } from './shared/components/header/header.component';
import { FooterComponent } from './shared/components/footer/footer.component';
import { TipsComponent } from './shared/components/tips/tips.component';
import { WeatherComponent } from './shared/components/weather/weather.component';
import { RpsComponent } from './shared/components/rps/rps.component';
import { GroceryComponent } from './shared/components/grocery/grocery.component';
import { HomeComponent } from './shared/components/home/home.component';
import { RagComponent } from './shared/components/rag/rag.component';


export const routes: Routes = [
  { path: 'rps', component: RpsComponent },
  { path: 'rag', component: RagComponent },
  { path: 'tips', component: TipsComponent },
  { path: 'grocery', component: GroceryComponent },
  { path: 'weather', component: WeatherComponent },
  { path: '', component: HomeComponent },
];


