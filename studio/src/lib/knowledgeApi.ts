// Knowledge upload client — multipart file upload (XHR for upload progress)
// plus the per-KB document listing. JSON endpoints reuse the shared `api`
// helper; the upload can't (FormData needs the browser-set multipart
// Content-Type with its boundary, and fetch has no upload progress).

import { api, ApiError } from "./api";

export interface UploadResult {
  document_id: string;
  chunks: number;
  characters: number;
  title: string | null;
  source_uri: string | null;
}

export interface KbDocument {
  document_id: string;
  title: string | null;
  source_uri: string | null;
  characters: number;
  chunks: number;
}

export const MAX_UPLOAD_BYTES = 15 * 1024 * 1024; // mirror the server cap
export const ALLOWED_EXTENSIONS = [".pdf", ".txt", ".md", ".csv"] as const;
export const ACCEPT_ATTR = ALLOWED_EXTENSIONS.join(",");

/** Client-side pre-flight: returns an error message, or null when uploadable. */
export function validateFile(file: File): string | null {
  const dot = file.name.lastIndexOf(".");
  const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
  if (!(ALLOWED_EXTENSIONS as readonly string[]).includes(ext)) {
    return `unsupported type ${ext || "(no extension)"} — pdf, txt, md or csv`;
  }
  if (file.size === 0) return "file is empty";
  if (file.size > MAX_UPLOAD_BYTES) {
    return `over the ${Math.floor(MAX_UPLOAD_BYTES / (1024 * 1024))} MB limit`;
  }
  return null;
}

/** Upload one file into a KB, reporting upload progress as a 0–1 fraction. */
export function uploadKbFile(
  kbId: string,
  file: File,
  onProgress?: (fraction: number) => void,
): Promise<UploadResult> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open(
      "POST",
      `/api/studio/kb/${encodeURIComponent(kbId)}/upload`,
    );
    xhr.responseType = "text";
    if (onProgress) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && e.total > 0) onProgress(e.loaded / e.total);
      };
    }
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* non-JSON body */
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body as UploadResult);
      } else {
        const detail = (body as { detail?: string } | null)?.detail;
        reject(
          new ApiError(
            xhr.status,
            detail ?? `${xhr.status} ${xhr.statusText || "upload failed"}`,
          ),
        );
      }
    };
    xhr.onerror = () => reject(new ApiError(0, "network error during upload"));
    xhr.onabort = () => reject(new ApiError(0, "upload cancelled"));
    const form = new FormData();
    form.append("file", file, file.name);
    xhr.send(form);
  });
}

/** The documents currently in a KB (in-memory store; backend KBs return []). */
export const listKbDocuments = (kbId: string) =>
  api.get<KbDocument[]>(`/kb/${encodeURIComponent(kbId)}/documents`);
