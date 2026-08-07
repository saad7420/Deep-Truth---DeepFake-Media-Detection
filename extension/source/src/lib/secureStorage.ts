// Obfuscation-grade encryption for sensitive values in chrome.storage.local
// using the Web Crypto API (AES-GCM). NOTE: the key is stored in the extension,
// so this protects against casual disk snooping, not a determined local
// attacker with full disk access.
const KEY_NAME = '__sec_key_v1__';

async function getKey(): Promise<CryptoKey> {
  const stored = await chrome.storage.local.get(KEY_NAME);
  if (stored[KEY_NAME]) {
    const raw = Uint8Array.from(atob(stored[KEY_NAME] as string), (c) => c.charCodeAt(0));
    return crypto.subtle.importKey('raw', raw, 'AES-GCM', true, ['encrypt', 'decrypt']);
  }
  const key = await crypto.subtle.generateKey({ name: 'AES-GCM', length: 256 }, true, ['encrypt', 'decrypt']);
  const raw = new Uint8Array(await crypto.subtle.exportKey('raw', key));
  await chrome.storage.local.set({ [KEY_NAME]: btoa(String.fromCharCode(...raw)) });
  return key;
}

export async function saveSecure(name: string, value: string): Promise<void> {
  const key = await getKey();
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const ct = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, new TextEncoder().encode(value));
  const bytes = new Uint8Array(iv.length + ct.byteLength);
  bytes.set(iv, 0);
  bytes.set(new Uint8Array(ct), iv.length);
  await chrome.storage.local.set({ [name]: btoa(String.fromCharCode(...bytes)) });
}

export async function loadSecure(name: string): Promise<string | null> {
  const stored = await chrome.storage.local.get(name);
  if (!stored[name]) return null;
  const key = await getKey();
  const bytes = Uint8Array.from(atob(stored[name] as string), (c) => c.charCodeAt(0));
  const iv = bytes.slice(0, 12);
  const ct = bytes.slice(12);
  const pt = await crypto.subtle.decrypt({ name: 'AES-GCM', iv }, key, ct);
  return new TextDecoder().decode(pt);
}