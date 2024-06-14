// rag-component.component.ts
import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { DecimalPipe, NgIf, NgFor } from '@angular/common';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { FormsModule } from '@angular/forms';



@Component({
  selector: 'rag',
  standalone: true,
  imports: [
  NgIf,
  NgFor,
  DecimalPipe,
  FormsModule,
  RouterOutlet,
  RouterLink,
  RouterLinkActive,
  ],
  templateUrl: './rag.component.html',
  styleUrls: ['./rag.component.css']
})
export class RagComponent {
  query: string = '';
  results: string[] = [];

  constructor(private http: HttpClient) {}

//  sendQuery() {
//    this.http.post<any>('http://localhost:8000/query', { query: this.query })
//      .subscribe(response => {
//        this.results = response.results;
//      });
//  }
//}

sendQuery() {
  const payload = { query: this.query };
  this.http.post<any>('http://localhost:8000/query', payload)
    .subscribe(
      response => {
        console.log(this.query);
        this.results = response.results;
      },
      error => {
        console.log(this.query);
        console.error('Error:', error);
        // Handle the error appropriately (e.g., display an error message to the user)
      }
    );
}




//    sendQuery() {
//      this.http.post<any>('http://localhost:8000/query', this.query, { headers: { 'Content-Type': 'application/json' } })
//        .subscribe(
//          response => {
//            console.log(this.query);
//            this.results = response.results;
//          },
//          error => {
//            console.log(this.query);
//            console.error('Error:', error);
//            // Handle the error appropriately (e.g., display an error message to the user)
//          }
//        );
//    }
}
