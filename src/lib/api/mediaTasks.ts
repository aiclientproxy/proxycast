import {
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_AUDIO_COMPLETE,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_AUDIO_CREATE,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_TRANSCRIPTION_CREATE,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_CANCEL,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_GET,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_IMAGE_COMPLETE,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_IMAGE_CREATE,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_LIST,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_VIDEO_CREATE,
  createAppServerClient,
} from "@/lib/api/appServer";
import type {
  CompleteAudioGenerationTaskArtifactRequest,
  CompleteImageGenerationTaskArtifactRequest,
  CreateAudioGenerationTaskArtifactRequest,
  CreateImageGenerationTaskArtifactRequest,
  CreateVideoGenerationTaskArtifactRequest,
  CreateTranscriptionTaskArtifactRequest,
  ListMediaTaskArtifactsOutput,
  ListMediaTaskArtifactsRequest,
  MediaTaskArtifactOutput,
  MediaTaskLookupRequest,
} from "./agentRuntime/mediaTaskTypes";

export {
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_AUDIO_COMPLETE,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_AUDIO_CREATE,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_TRANSCRIPTION_CREATE,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_CANCEL,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_GET,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_IMAGE_COMPLETE,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_IMAGE_CREATE,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_LIST,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_VIDEO_CREATE,
};

export async function createImageGenerationTaskArtifact(
  request: CreateImageGenerationTaskArtifactRequest,
): Promise<MediaTaskArtifactOutput> {
  return (await createAppServerClient().createImageMediaTaskArtifact(request))
    .result as MediaTaskArtifactOutput;
}

export async function createAudioGenerationTaskArtifact(
  request: CreateAudioGenerationTaskArtifactRequest,
): Promise<MediaTaskArtifactOutput> {
  return (await createAppServerClient().createAudioMediaTaskArtifact(request))
    .result as MediaTaskArtifactOutput;
}

export async function createVideoGenerationTaskArtifact(
  request: CreateVideoGenerationTaskArtifactRequest,
): Promise<MediaTaskArtifactOutput> {
  return (await createAppServerClient().createVideoMediaTaskArtifact(request))
    .result as MediaTaskArtifactOutput;
}

export async function createTranscriptionTaskArtifact(
  request: CreateTranscriptionTaskArtifactRequest,
): Promise<MediaTaskArtifactOutput> {
  return (
    await createAppServerClient().createTranscriptionMediaTaskArtifact(request)
  ).result as MediaTaskArtifactOutput;
}

export async function completeImageGenerationTaskArtifact(
  request: CompleteImageGenerationTaskArtifactRequest,
): Promise<MediaTaskArtifactOutput> {
  return (await createAppServerClient().completeImageMediaTaskArtifact(request))
    .result as MediaTaskArtifactOutput;
}

export async function completeAudioGenerationTaskArtifact(
  request: CompleteAudioGenerationTaskArtifactRequest,
): Promise<MediaTaskArtifactOutput> {
  return (await createAppServerClient().completeAudioMediaTaskArtifact(request))
    .result as MediaTaskArtifactOutput;
}

export async function getMediaTaskArtifact(
  request: MediaTaskLookupRequest,
): Promise<MediaTaskArtifactOutput> {
  return (await createAppServerClient().getMediaTaskArtifact(request))
    .result as MediaTaskArtifactOutput;
}

export async function listMediaTaskArtifacts(
  request: ListMediaTaskArtifactsRequest,
): Promise<ListMediaTaskArtifactsOutput> {
  return (await createAppServerClient().listMediaTaskArtifacts(request))
    .result as ListMediaTaskArtifactsOutput;
}

export async function cancelMediaTaskArtifact(
  request: MediaTaskLookupRequest,
): Promise<MediaTaskArtifactOutput> {
  return (await createAppServerClient().cancelMediaTaskArtifact(request))
    .result as MediaTaskArtifactOutput;
}
