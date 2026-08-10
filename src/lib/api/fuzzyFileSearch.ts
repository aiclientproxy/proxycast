import {
  AppServerClient,
  type AppServerFuzzyFileSearchResult,
  type AppServerRequestOptions,
} from "@/lib/api/appServer";

export type ProjectFileSearchResult = AppServerFuzzyFileSearchResult;

export type FuzzyFileSearchClient = Pick<AppServerClient, "searchFiles">;

export async function searchProjectFiles(
  params: {
    query: string;
    rootPath: string;
    cancellationToken: string;
  },
  options: AppServerRequestOptions = {},
  client: FuzzyFileSearchClient = new AppServerClient(),
): Promise<ProjectFileSearchResult[]> {
  const response = await client.searchFiles(
    {
      query: params.query,
      roots: [params.rootPath],
      cancellationToken: params.cancellationToken,
    },
    options,
  );
  return response.result.files;
}
