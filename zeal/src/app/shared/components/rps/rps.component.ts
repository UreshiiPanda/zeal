import { Component } from '@angular/core';
import { faGem, faPaperPlane, faScissors } from '@fortawesome/free-solid-svg-icons';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';
import { FormsModule } from '@angular/forms';
import { DecimalPipe } from '@angular/common';



@Component({
  selector: 'rps',
  standalone: true,
  imports: [
    FontAwesomeModule,
    DecimalPipe,
    FormsModule,
    RouterOutlet,
    RouterLink,
    RouterLinkActive,

  ],
  templateUrl: './rps.component.html',
  styleUrl: './rps.component.css'
})
export class RpsComponent {
  faGem = faGem;
  faPaperPlane = faPaperPlane;
  faScissors = faScissors;

}
