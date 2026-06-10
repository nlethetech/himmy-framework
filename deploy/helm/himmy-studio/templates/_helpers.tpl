{{/*
Expand the name of the chart.
*/}}
{{- define "himmy-studio.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this
(by the DNS naming spec).
*/}}
{{- define "himmy-studio.fullname" -}}
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
{{- define "himmy-studio.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "himmy-studio.labels" -}}
helm.sh/chart: {{ include "himmy-studio.chart" . }}
{{ include "himmy-studio.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "himmy-studio.selectorLabels" -}}
app.kubernetes.io/name: {{ include "himmy-studio.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Name of the PVC backing /app/.himmy.
Honors persistence.existingClaim when set.
*/}}
{{- define "himmy-studio.pvcName" -}}
{{- if .Values.persistence.existingClaim }}
{{- .Values.persistence.existingClaim }}
{{- else }}
{{- printf "%s-state" (include "himmy-studio.fullname" .) }}
{{- end }}
{{- end }}

{{/*
himmy-studio.allowHosts
Computes the value for HIMMY_STUDIO_ALLOW_HOSTS as a comma-separated string by
UNIONING studio.allowHosts with every ingress.hosts[*].host (when ingress is
enabled). This guarantees an enabled ingress can never 403 itself on the
loopback guard. Duplicate and empty hosts are removed. Order: explicit
allowHosts first, then ingress hosts.
*/}}
{{- define "himmy-studio.allowHosts" -}}
{{- $hosts := list -}}
{{- range .Values.studio.allowHosts -}}
{{- $h := . | toString | trim -}}
{{- if $h -}}
{{- $hosts = append $hosts $h -}}
{{- end -}}
{{- end -}}
{{- if .Values.ingress.enabled -}}
{{- range .Values.ingress.hosts -}}
{{- $h := .host | toString | trim -}}
{{- if $h -}}
{{- $hosts = append $hosts $h -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- $hosts | uniq | join "," -}}
{{- end }}
