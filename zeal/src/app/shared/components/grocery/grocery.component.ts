import { Component, input } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';
import { faPenToSquare } from '@fortawesome/free-solid-svg-icons';
import { faTrash } from '@fortawesome/free-solid-svg-icons';
import { FormsModule } from '@angular/forms';
import { DecimalPipe, NgFor } from '@angular/common';


@Component({
  selector: 'grocery',
  standalone: true,
  imports: [
    NgFor,
    FontAwesomeModule,
    DecimalPipe,
    FormsModule,
    RouterOutlet,
    RouterLink,
    RouterLinkActive,
  ],
  templateUrl: './grocery.component.html',
  styleUrl: './grocery.component.css'
})

export class GroceryComponent {
  faPenToSquare = faPenToSquare;
  faTrash = faTrash;
  newItem: string = '';
  items: string[] = [];

  addItem() {
    if (this.newItem.trim() !== '') {
      this.items.unshift(this.newItem.trim());
      this.newItem = '';
    }
  }

  editItem(index: number) {
    const item = this.items[index];
    const newItem = prompt('Edit item', item);
    if (newItem !== null && newItem.trim() !== '') {
      this.items[index] = newItem.trim();
    }
  }

  deleteItem(index: number) {
    this.items.splice(index, 1);
  }

  clearItems() {
    this.items = [];
  }
}
