use app_server_protocol::ProjectGitCommit;
use app_server_protocol::ProjectGitCommitListResponse;
use app_server_protocol::ProjectGitDiffResponse;
use app_server_protocol::ProjectGitStatusResponse;
use app_server_protocol::ProjectGitWorktreeCreateResponse;

pub(super) fn project_git_status_from_service(
    status: lime_services::project_git_service::ProjectGitStatus,
) -> ProjectGitStatusResponse {
    ProjectGitStatusResponse {
        root_path: status.root_path,
        repository_root: status.repository_root,
        has_git_repository: status.has_git_repository,
        current_branch: status.current_branch,
        branches: status.branches,
        uncommitted_file_count: status.uncommitted_file_count,
    }
}

pub(super) fn project_git_diff_from_service(
    diff: lime_services::project_git_service::ProjectGitDiff,
) -> ProjectGitDiffResponse {
    ProjectGitDiffResponse {
        root_path: diff.root_path,
        repository_root: diff.repository_root,
        has_git_repository: diff.has_git_repository,
        current_ref: diff.current_ref,
        comparison_base_ref: diff.comparison_base_ref,
        patch: diff.patch,
        uncommitted_file_count: diff.uncommitted_file_count,
    }
}

pub(super) fn project_git_commit_list_from_service(
    list: lime_services::project_git_service::ProjectGitCommitList,
) -> ProjectGitCommitListResponse {
    ProjectGitCommitListResponse {
        root_path: list.root_path,
        repository_root: list.repository_root,
        has_git_repository: list.has_git_repository,
        commits: list
            .commits
            .into_iter()
            .map(project_git_commit_from_service)
            .collect(),
    }
}

fn project_git_commit_from_service(
    commit: lime_services::project_git_service::ProjectGitCommit,
) -> ProjectGitCommit {
    ProjectGitCommit {
        sha: commit.sha,
        short_sha: commit.short_sha,
        subject: commit.subject,
        author_name: commit.author_name,
        author_email: commit.author_email,
        committed_at: commit.committed_at,
    }
}

pub(super) fn project_git_worktree_from_service(
    worktree: lime_services::project_git_service::ProjectGitWorktree,
) -> ProjectGitWorktreeCreateResponse {
    ProjectGitWorktreeCreateResponse {
        worktree_path: worktree.worktree_path,
        branch: worktree.branch,
        status: project_git_status_from_service(worktree.status),
    }
}
