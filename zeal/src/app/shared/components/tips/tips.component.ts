import { Component, input } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';
import { faDollarSign } from '@fortawesome/free-solid-svg-icons';
import { FormsModule } from '@angular/forms';
import { DecimalPipe } from '@angular/common';


@Component({
  selector: 'tips',
  standalone: true,
  imports: [
    FontAwesomeModule,
    DecimalPipe,
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
  tipAmount: number = 0.00;
  totalBillAmount: number = 0.00;
  totalBill: number = 0;

  onTipChange(event: any) {
    this.tipValue = event.target.value;
    this.calculateTipAmount();
    this.calculateTotalBillAmount();
  }

  calculateTipAmount() {
    this.tipAmount = (this.totalBill * this.tipValue) / 100;
  }

  calculateTotalBillAmount() {
    this.totalBillAmount = this.totalBill + this.tipAmount;
  }

}


