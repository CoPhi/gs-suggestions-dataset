import { inject, Injectable, signal } from '@angular/core';
import { SwPush } from '@angular/service-worker';

export interface AppAlert {
  id: string;
  message: string;
  type: 'success' | 'danger' | 'warning' | 'info';
}

@Injectable({
  providedIn: 'root'
})
export class NotificationService {
  readonly alerts = signal<AppAlert[]>([]);

  showAlert(message: string, type: 'success' | 'danger' | 'warning' | 'info' = 'info', durationMs = 5000) {
    const id = Math.random().toString(36).substring(2, 9);
    const newAlert: AppAlert = { id, message, type };
    this.alerts.update(current => [...current, newAlert]);

    if (durationMs > 0) {
      setTimeout(() => {
        this.removeAlert(id);
      }, durationMs);
    }
  }

  showSuccess(message: string, durationMs = 5000) {
    this.showAlert(message, 'success', durationMs);
  }

  showDanger(message: string, durationMs = 5000) {
    this.showAlert(message, 'danger', durationMs);
  }

  showWarning(message: string, durationMs = 5000) {
    this.showAlert(message, 'warning', durationMs);
  }

  removeAlert(id: string) {
    this.alerts.update(current => current.filter(alert => alert.id !== id));
  }

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
    if ("Notification" in window && Notification.permission === "granted") {
      return new Notification("Generazione completata 🎉", {
        body: "I suggerimenti sono stati generati con successo.",
      });
    }
    return null;
  }
}