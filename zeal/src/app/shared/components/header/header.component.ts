import { Component } from '@angular/core';
import { RouterModule, RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';
import { faSailboat } from '@fortawesome/free-solid-svg-icons';


@Component({
  selector: 'header',
  standalone: true,
  imports: [
    FontAwesomeModule,
    RouterModule,
    RouterOutlet,
    RouterLink,
    RouterLinkActive,
  ],
  templateUrl: './header.component.html',
  styleUrl: './header.component.css'
})
export class HeaderComponent {
  faSailboat = faSailboat;
}
