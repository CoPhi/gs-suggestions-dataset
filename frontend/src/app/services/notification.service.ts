import { inject, Injectable } from '@angular/core';
import { SwPush } from '@angular/service-worker';

@Injectable({
  providedIn: 'root'
})
export class NotificationService {

  enableNotifications() {
    if ("Notification" in window) {
      Notification.requestPermission().then(permission => {
        if (permission === "granted") {
          console.log("Notifiche abilitate!");
        } else {
          console.log("Notifiche non abilitate");
        }
      });
    }
  }

  showLocalNotification() {
      return new Notification("Generazione completata 🎉", {
            body: "I suggerimenti sono stati generati con successo.",
          });
        }
}