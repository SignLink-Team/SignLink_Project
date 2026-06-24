/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export type ViewState = 'main' | 'history' | 'login';

export interface TranslationLog {
  id: string;
  text: string;
  timestamp: string;
  category?: string;
  isCustom?: boolean;
}

export interface UserState {
  isLoggedIn: boolean;
  email: string | null;
  name: string | null;
}

export interface GesturePreset {
  id: string;
  gestureName: string;
  translationText: string;
  category: string;
  description: string;
}
