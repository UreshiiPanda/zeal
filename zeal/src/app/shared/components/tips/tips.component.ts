import { Component, input } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';
import { faDollarSign } from '@fortawesome/free-solid-svg-icons';
import { FormsModule } from '@angular/forms';


@Component({
  selector: 'tips',
  standalone: true,
  imports: [
    FontAwesomeModule,
    FormsModule,
    RouterOutlet,
    RouterLink,
    RouterLinkActive,
  ],
  templateUrl: './tips.component.html',
  styleUrl: './tips.component.css'
})

export class TipsComponent {
  faDollarSign = faDollarSign;
  tipValue: number = 0;
  totalBill: number = 0;
  onTipChange(event: any) {
    this.tipValue = event.target.value;
  }
}

