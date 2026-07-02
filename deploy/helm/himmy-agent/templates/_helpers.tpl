{{/*
Expand the name of the chart.
*/}}
{{- define "himmy-agent.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this
(by the DNS naming spec).
*/}}
{{- define "himmy-agent.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "himmy-agent.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "himmy-agent.labels" -}}
helm.sh/chart: {{ include "himmy-agent.chart" . }}
{{ include "himmy-agent.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels. `component` is threaded in per-Deployment so the api and worker
pods carry distinct selectors while sharing the instance labels.
*/}}
{{- define "himmy-agent.selectorLabels" -}}
app.kubernetes.io/name: {{ include "himmy-agent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Name of the PVC backing /app/.himmy (shared by api + worker).
Honors persistence.existingClaim when set.
*/}}
{{- define "himmy-agent.pvcName" -}}
{{- if .Values.persistence.existingClaim }}
{{- .Values.persistence.existingClaim }}
{{- else }}
{{- printf "%s-state" (include "himmy-agent.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Name of the ConfigMap holding the agent.yaml.
Honors agent.existingConfigMap when set.
*/}}
{{- define "himmy-agent.configMapName" -}}
{{- if .Values.agent.existingConfigMap }}
{{- .Values.agent.existingConfigMap }}
{{- else }}
{{- printf "%s-spec" (include "himmy-agent.fullname" .) }}
{{- end }}
{{- end }}

{{/*
The ConfigMap key that holds the agent.yaml content.
*/}}
{{- define "himmy-agent.configMapKey" -}}
{{- if .Values.agent.existingConfigMap }}
{{- .Values.agent.existingConfigMapKey }}
{{- else }}
{{- "agent.yaml" }}
{{- end }}
{{- end }}

{{/*
himmy-agent.commonEnv
The env block shared by api + worker: durable store on the shared PVC, optional
Postgres entity store, file-based secrets, auth mode, and free-form passthrough.
Rendered as a YAML list; callers `nindent` it under `env:`.
*/}}
{{- define "himmy-agent.commonEnv" -}}
- name: HIMMY_DURABLE_STORAGE
  value: "1"
- name: HIMMY_STORE_PATH
  value: /app/.himmy/storage.db
{{- if .Values.secrets.existingSecret }}
- name: HIMMY_SECRETS
  value: "file"
- name: HIMMY_SECRETS_DIR
  value: {{ .Values.secrets.mountPath | quote }}
{{- end }}
{{- if .Values.database.existingSecret }}
- name: HIMMY_DATABASE_URL
  valueFrom:
    secretKeyRef:
      name: {{ .Values.database.existingSecret | quote }}
      key: {{ .Values.database.existingSecretKey | quote }}
{{- else if .Values.database.url }}
- name: HIMMY_DATABASE_URL
  value: {{ .Values.database.url | quote }}
{{- end }}
{{- if .Values.auth.mode }}
- name: HIMMY_AUTH_MODE
  value: {{ .Values.auth.mode | quote }}
{{- end }}
{{- if .Values.auth.apiKeysFile }}
- name: HIMMY_API_KEYS_FILE
  value: {{ .Values.auth.apiKeysFile | quote }}
{{- end }}
{{- range $key, $value := .Values.env }}
- name: {{ $key }}
  value: {{ $value | quote }}
{{- end }}
{{- end }}

{{/*
himmy-agent.volumes
The pod volumes shared by api + worker: the agent.yaml ConfigMap, the RWO state
PVC, and (optionally) the file-secrets Secret. Callers `nindent` it under `volumes:`.
*/}}
{{- define "himmy-agent.volumes" -}}
- name: agent-spec
  configMap:
    name: {{ include "himmy-agent.configMapName" . }}
    items:
      - key: {{ include "himmy-agent.configMapKey" . }}
        path: agent.yaml
- name: state
  {{- if .Values.persistence.enabled }}
  persistentVolumeClaim:
    claimName: {{ include "himmy-agent.pvcName" . }}
  {{- else }}
  emptyDir: {}
  {{- end }}
{{- if .Values.secrets.existingSecret }}
- name: secrets
  secret:
    secretName: {{ .Values.secrets.existingSecret }}
{{- end }}
{{- end }}

{{/*
himmy-agent.volumeMounts
The container mounts matching himmy-agent.volumes. The agent.yaml is mounted as a
single subPath file (read-only) at the stable CMD path /app/agent.yaml.
*/}}
{{- define "himmy-agent.volumeMounts" -}}
- name: agent-spec
  mountPath: /app/agent.yaml
  subPath: agent.yaml
  readOnly: true
- name: state
  mountPath: /app/.himmy
{{- if .Values.secrets.existingSecret }}
- name: secrets
  mountPath: {{ .Values.secrets.mountPath }}
  readOnly: true
{{- end }}
{{- end }}
