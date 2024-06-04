import { Component } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';
import { faDollarSign } from '@fortawesome/free-solid-svg-icons';


@Component({
  selector: 'tips',
  standalone: true,
  imports: [
    FontAwesomeModule,
    RouterOutlet,
    RouterLink,
    RouterLinkActive,
  ],
  templateUrl: './tips.component.html',
  styleUrl: './tips.component.css'
})

export class TipsComponent {
  faDollarSign = faDollarSign;
}
