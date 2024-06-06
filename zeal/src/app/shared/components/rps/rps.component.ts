import { Component } from '@angular/core';
import { faGem, faPaperPlane, faScissors } from '@fortawesome/free-solid-svg-icons';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';
import { FormsModule } from '@angular/forms';
import { DecimalPipe, NgIf } from '@angular/common';


@Component({
  selector: 'rps',
  standalone: true,
  imports: [
    NgIf,
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

  userChoice: string = '';
  computerChoice: string = '';
  result: string = '';

  makeChoice(choice: string) {
    this.userChoice = choice;
    this.computerChoice = this.getComputerChoice();
    this.result = this.determineResult(choice, this.computerChoice);
  }

  getComputerChoice(): string {
    const choices = ['Rock', 'Paper', 'Scissors'];
    const randomIndex = Math.floor(Math.random() * choices.length);
    return choices[randomIndex];
  }

  determineResult(userChoice: string, computerChoice: string): string {
    if (userChoice === computerChoice) {
      return 'tie';
    } else if (
      (userChoice === 'Rock' && computerChoice === 'Scissors') ||
      (userChoice === 'Paper' && computerChoice === 'Rock') ||
      (userChoice === 'Scissors' && computerChoice === 'Paper')
    ) {
      return 'win';
    } else {
      return 'lose';
    }
  }

  resetGame() {
    this.userChoice = '';
    this.computerChoice = '';
    this.result = '';
  }
}
